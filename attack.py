import numpy as np
import imageio
import torch
import lpips
from keras.applications.resnet50 import decode_predictions
from skimage.measure import compare_ssim, compare_psnr
from image_similarity_measures.quality_metrics import fsim

count = 0

def compute_lpips(loss_fn, x1, x2, image_size=224):
    x1 = x1.reshape((3, image_size, image_size))
    x1 = np.expand_dims(x1, axis=0).astype(np.float32)
    x2 = x2.reshape((3, image_size, image_size))
    x2 = np.expand_dims(x2, axis=0)
    
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    d = loss_fn(x1, x2)

    return d.item()


def attack(model,
    sample,
    method = 'idba',
    mask = None,
    clip_max = 1,
    clip_min = 0,
    num_iterations = 40,
    gamma = 1.0,
    target_label = None,
    target_image = None,
    max_num_evals = 1e4,
    init_num_evals = 100,
    verbose = True, 
    image_size=224,
    max_query=15000,
    rv_generator=None,):
    
    
    # Initialize the number of queries.
    global count
    count = 0

    # Set parameters
    loss_fn = lpips.LPIPS(net='alex')
    original_label = np.argmax(model.predict(np.expand_dims(sample, 0)))
    print("original label: ", decode_predictions(model.predict(np.expand_dims(sample, 0)), top=1)[0])
    print("target image label: ", decode_predictions(model.predict(np.expand_dims(target_image, 0)), top=1)[0])
    params = {'clip_max': clip_max, 'clip_min': clip_min,
                'shape': sample.shape,
                'original_label': original_label,
                'target_label': target_label,
                'target_image': target_image,
                'num_iterations': num_iterations,
                'gamma': gamma,
                'd': int(np.prod(sample.shape)),
                'max_num_evals': max_num_evals,
                'init_num_evals': init_num_evals,
                'verbose': verbose,
                'mask': mask,
                'method': method,
                'max_query': max_query,
                'rv_generator': rv_generator,
                }

    # Set binary search threshold.
    params['theta'] = params['gamma'] / (np.sqrt(params['d']) * params['d'])

    # Initialize.
    perturbed = params['target_image']

    # Project the initialization to the boundary.
    perturbed, dist_post_update = binary_search_batch(sample,
        np.expand_dims(perturbed, 0),
        model,
        params)
    dist = compute_distance(perturbed, sample)

    for j in np.arange(params['num_iterations']):
        params['cur_iter'] = j + 1
        
        # Choose delta.
        delta = select_delta(params, dist_post_update)

        # Choose number of evaluations.
        num_evals = int(params['init_num_evals'] * np.sqrt(j+1))
        num_evals = int(min([num_evals, params['max_num_evals']]))

        # approximate gradient.
        gradf = approximate_gradient(model, perturbed, num_evals,
            delta, params)
        update = gradf

        # search step size.
        epsilon = geometric_progression_for_stepsize(perturbed,
            update, dist, model, params)

        # Update the sample.
        perturbed = clip_image(perturbed + epsilon * update,
            clip_min, clip_max)
        
        if params['method'] == 'idba':
            # Binary search in high-frequency components to return to the boundary.
            perturbed, dist_post_update = binary_search_batch_mask(sample,
                perturbed[None], model, params)    

        # Binary search to return to the boundary.
        perturbed, dist_post_update = binary_search_batch(sample,
            perturbed[None], model, params)

        ssim_j = compare_ssim(sample, perturbed, data_range=1, multichannel=True)
        psnr_j = compare_psnr(sample, perturbed, data_range=1)
        fsim_j = fsim(sample, perturbed)
        dist_j = compute_distance(perturbed, sample)
        lpips_j = compute_lpips(loss_fn, perturbed, sample, image_size=image_size)
        imageio.imwrite("./results/%s/adv_img_%d.png" % (method, count), (perturbed * 255.0).astype(np.uint8))
        print("iter: %d, q: %d, ssim: %.4f, psnr: %.4f, fsim: %.4f, lpips: %.4f, dist: %.4f" % (j + 1, count, ssim_j, psnr_j, fsim_j, lpips_j, dist_j))
            
        if count >= params['max_query']:
            break
        
    return perturbed

def decision_function(model, images, params):
    """
    Decision function output 1 on the desired side of the boundary,
    0 otherwise.
    """
    global count
    images = clip_image(images, params['clip_min'], params['clip_max'])
    prob = model.predict(images)
    count += images.shape[0]
    return np.argmax(prob, axis = 1) == params['target_label']

def clip_image(image, clip_min, clip_max):
    # Clip an image, or an image batch, with upper and lower threshold.
    return np.minimum(np.maximum(clip_min, image), clip_max)


def compute_distance(x_ori, x_pert):
    # Compute the distance between two images.
    return np.linalg.norm(x_ori - x_pert)


def approximate_gradient(model, sample, num_evals, delta, params):
    clip_max, clip_min = params['clip_max'], params['clip_min']

    # Generate random vectors.
    noise_shape = [num_evals] + list(params['shape'])
    if params['method'] == 'idba':
        rv = np.random.randn(*noise_shape) * params['mask']
    elif params['method'] == 'qeba':
        rv = params['rv_generator'].generate_ps(sample, num_evals)
    else:
        rv = np.random.randn(*noise_shape)

    rv = rv / np.sqrt(np.sum(rv ** 2, axis = (1,2,3), keepdims = True))
    perturbed = sample + delta * rv
    perturbed = clip_image(perturbed, clip_min, clip_max)
    rv = (perturbed - sample) / delta

    # query the model.
    decisions = decision_function(model, perturbed, params)
    decision_shape = [len(decisions)] + [1] * len(params['shape'])
    fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0

    # Baseline subtraction (when fval differs)
    if np.mean(fval) == 1.0: # label changes.
        gradf = np.mean(rv, axis = 0)
    elif np.mean(fval) == -1.0: # label not change.
        gradf = - np.mean(rv, axis = 0)
    else:
        fval -= np.mean(fval)
        gradf = np.mean(fval * rv, axis = 0)

    # Get the gradient direction.
    gradf = gradf / np.linalg.norm(gradf)

    return gradf


def project(original_image, perturbed_images, highs, params):
    alphas = 1 - highs
    alphas_shape = [len(alphas)] + [1] * len(params['shape'])
    alphas = alphas.reshape(alphas_shape)
    return alphas * original_image + (1 - alphas) * perturbed_images


def project_mask(original_image, perturbed_images, highs, params):
    alphas = 1 - highs
    alphas_shape = [len(alphas)] + [1] * len(params['shape'])
    alphas = alphas.reshape(alphas_shape)
    return perturbed_images + alphas * (original_image - perturbed_images) * (1 - params['mask'])

def binary_search_batch(original_image, perturbed_images, model, params):
    """ Binary search to approach the boundar. """

    # Compute distance between each of perturbed image and original image.
    dists_post_update = np.array([
            compute_distance(
                original_image,
                perturbed_image,
            )
            for perturbed_image in perturbed_images])

    # Choose upper thresholds in binary searchs.
    highs = np.ones(len(perturbed_images))
    thresholds = params['theta']

    lows = np.zeros(len(perturbed_images))

    # Call recursive function.
    while np.max((highs - lows) / thresholds) > 1:
        # projection to mids.
        mids = (highs + lows) / 2.0
        mid_images = project(original_image, perturbed_images, mids, params)

        # Update highs and lows based on model decisions.
        decisions = decision_function(model, mid_images, params)
        lows = np.where(decisions == 0, mids, lows)
        highs = np.where(decisions == 1, mids, highs)

    out_images = project(original_image, perturbed_images, highs, params)

    # Compute distance of the output image to select the best choice.
    # (only used when stepsize_search is grid_search.)
    dists = np.array([
        compute_distance(
            original_image,
            out_image,
        )
        for out_image in out_images])
    idx = np.argmin(dists)

    dist = dists_post_update[idx]
    out_image = out_images[idx]
    return out_image, dist

def binary_search_batch_mask(original_image, perturbed_images, model, params):
    """ Binary search to approach the boundar. """

    # Compute distance between each of perturbed image and original image.
    dists_post_update = np.array([
            compute_distance(
                original_image,
                perturbed_image,
            )
            for perturbed_image in perturbed_images])

    # Choose upper thresholds in binary searchs.
    highs = np.ones(len(perturbed_images))
    # thresholds = params['theta']
    thresholds = params['theta'] * ((np.count_nonzero(params['mask']) / params['d']) ** (-3/2))

    lows = np.zeros(len(perturbed_images))

    # Call recursive function.
    while np.max((highs - lows) / thresholds) > 1:
        # projection to mids.
        mids = (highs + lows) / 2.0
        mid_images = project_mask(original_image, perturbed_images, mids, params)

        # Update highs and lows based on model decisions.
        decisions = decision_function(model, mid_images, params)
        lows = np.where(decisions == 0, mids, lows)
        highs = np.where(decisions == 1, mids, highs)

    out_images = project_mask(original_image, perturbed_images, highs, params)

    # Compute distance of the output image to select the best choice.
    # (only used when stepsize_search is grid_search.)
    dists = np.array([
        compute_distance(
            original_image,
            out_image,
        )
        for out_image in out_images])
    idx = np.argmin(dists)

    dist = dists_post_update[idx]
    out_image = out_images[idx]
    return out_image, dist


def geometric_progression_for_stepsize(x, update, dist, model, params):
    """
    Geometric progression to search for stepsize.
    Keep decreasing stepsize by half until reaching 
    the desired side of the boundary,
    """
    if params['method'] == 'idba':
        epsilon = dist * np.sqrt(np.count_nonzero(params['mask']) / params['d']) / np.sqrt(params['cur_iter'])
    else:
        epsilon = dist / np.sqrt(params['cur_iter'])

    def phi(epsilon):
        new = x + epsilon * update
        success = decision_function(model, new[None], params)
        return success

    while not phi(epsilon):
        epsilon /= 2.0

    return epsilon

def select_delta(params, dist_post_update):
    """ 
    Choose the delta at the scale of distance 
    between x and perturbed sample. 

    """
    if params['cur_iter'] == 1:
        delta = 0.1 * (params['clip_max'] - params['clip_min'])
    else:
        delta = np.sqrt(params['d']) * params['theta'] * dist_post_update

    return delta
