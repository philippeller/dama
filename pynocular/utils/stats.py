"""Collection of some statistics functions"""

import numpy as np

__license__ = '''Copyright 2020 Philipp Eller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

def weighted_quantile(samples, weights, q, unify=False, interp='correct', method=7):
    '''
    weighted quantile function
    
    Parameters:
    ----------
    samples : array
        points
    weights : array
        weights of points
    q : array
        quantiles to compute
    unify : bool (optional)
        add up weights of identical points, resulting in single points
    interp : string
        interpolation method, choice of ['new', 'const', 'linear']
        
    Notes:
    ------
    weights must be > 0 and < inf
    
    '''

    
    samples = np.asarray(samples)
    weights = np.asarray(weights)
    
    assert np.alltrue(weights > 0), 'Weights must be > 0'
    assert np.alltrue(weights < np.inf), 'Weights must be < inf'

    q = np.asarray(q)

    if unify:
        unique_samples, weight_idx = np.unique(samples, return_inverse=True)
        unique_weights = np.zeros(len(unique_samples))
        for i, j in enumerate(weight_idx):
            unique_weights[j] += weights[i]
        samples = unique_samples
        weights = unique_weights
        
    # sort
    sorted_idx = np.argsort(samples)
    samples = samples[sorted_idx]
    weights = weights[sorted_idx]

    
    # arithmetic mean
    widths = 0.5 * (weights[:-1] + weights[1:])
    # geometric mean
    #widths = np.sqrt(weights[:-1] * weights[1:])
    # harmonic mean
    #widths = 1/ (1/weights[:-1] + 1/weights[1:])
    widths /= np.sum(widths)
 

    edges = np.concatenate([[0.], np.cumsum(weights)])
    
    if method == 7:

        # compute weighted edges and midpoints
        edges -= 0.5 * weights[0]
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        total_width = sum(weights) - weights[0]/2 - weights[-1]/2 
        edges /= total_width
        midpoints = midpoints / total_width
        
    
    elif method == 5:
        
        edges /= np.sum(weights)
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        
        
    if interp == 'expand':

        delta = (weights[1:] - weights[:-1])
        #weight_ratios = (weights[:-1] - weights[1:]) / (weights[:-1] + weights[1:])# lower / upper
        #weight_ratios = delta / (1 + np.abs(delta))
        weight_ratios = delta / np.maximum(weights[1:] , weights[:-1])
    
        widths = np.diff(midpoints)
        
        new_x = np.zeros_like(weight_ratios)
        new_f = np.zeros_like(weight_ratios)
        
        mask = weight_ratios > 0
        # i.e. lower bigger
        new_x[mask] = midpoints[1:][mask] - weight_ratios[mask] * widths[mask] #* 2/3
        new_f[mask] = samples[1:][mask]
        
        mask = ~mask
        # i.e. upper bigger
        new_x[mask] = midpoints[:-1][mask] - weight_ratios[mask] * widths[mask] #* 2/3
        new_f[mask] = samples[:-1][mask]        
        
        
        xp = np.empty(2 * len(midpoints) -1)
        fp = np.empty_like(xp)
        
        xp[::2] = midpoints
        fp[::2] = samples
        xp[1::2] = new_x
        fp[1::2] = new_f

        return np.interp(q, xp, fp)      

    
    if interp == 'weighted fraction':    
        d = np.digitize(q, midpoints, right=True)

        # include left
        d[d==0] = 1
        d -= 1

        # include right
        d[d==len(midpoints) - 1] = len(midpoints) - 2

        # x = partial distance between points
        x = (q - midpoints[d]) / (midpoints[d+1] - midpoints[d])

        out = ((1-x)*weights[d]*samples[d] + x*weights[d+1]*samples[d+1])/((1-x)*weights[d] + x*weights[d+1])
        
        out[q > midpoints[-1]] = samples[-1]
        out[q < midpoints[0]] = samples[0]
            
        return out
            
    if interp == 'correct':    
        d = np.digitize(q, midpoints, right=True)

        # include left
        d[d==0] = 1
        d -= 1

        # include right
        d[d==len(midpoints) - 1] = len(midpoints) - 2

        linear = np.interp(q, midpoints, samples)
        
        # partial quantiles
        pq = (q - midpoints[d]) / (midpoints[d+1] - midpoints[d])
        
        x0 = samples[d]
        x1 = samples[d+1]
        w0 = weights[d]
        w1 = weights[d+1]
        
        
        out = (-np.sqrt(((2 *x0 *w1)/((x0 - x1)**2 *(w1 + w0)) - (2 *w0 *x1)/((x0 - x1)**2 * (w1 + w0)))**2 - 4 *(w0/((x0 - x1)**2 * (w1 + w0)) - w1/((x0 - x1)**2 * (w1 + w0))) *(pq - (w0 *x0**2)/((x0 - x1)**2 *(w1 + w0)) - (x0**2 *w1)/((x0 - x1)**2 *(w1 + w0)) + (2 *w0 *x0 *x1)/((x0 - x1)**2 *(w1 + w0)))) - (2 *x0 *w1)/((x0 - x1)**2 *(w1 + w0)) + (2 *w0 *x1)/((x0 - x1)**2 *(w1 + w0)))/(2 *(w0/((x0 - x1)**2 *(w1 + w0)) - w1/((x0 - x1)**2 *(w1 + w0))))

        #return out
        mask = w0 == w1
        out[mask] = linear[mask]
        
        out[q > midpoints[-1]] = samples[-1]
        out[q < midpoints[0]] = samples[0]
    
        return out
    
    
    if interp == 'linear':
        return np.interp(q, midpoints, samples)
        
    if interp == 'nearest':
        d = np.digitize(q, edges, right=True)
        # include left
        d[d==0] = 1
        d -= 1
        d = np.clip(d, a_min=0, a_max=len(samples)-1)
        return samples[d]
        
    if interp == 'lower':
        d = np.digitize(q, midpoints, right=False)
        d -= 1
        d[d<0] = 0
        return samples[d]

    elif interp == 'higher':
        d = np.digitize(q, midpoints, right=True)
        d[d>=len(samples)] = len(samples) - 1
        return samples[d]
        
    elif interp == 'midpoint':
        sample_midpoints = (weights[:-1] * samples[:-1] + weights[1:] * samples[1:]) / (weights[:-1] + weights[1:])
        
        d = np.digitize(q, midpoints, right=True)
        d[d==0] = 1
        d-=1
        d[d==len(midpoints) - 1] = len(midpoints) - 2 
            
        out = sample_midpoints[d]
        # edge cases
        if q[0] == 0:
            out[0] = samples[0]
        if q[-1] == 1:
            out[-1] = samples[-1]
        return out
    
    elif interp == 'corners':
        sample_midpoints = (weights[:-1] * samples[:-1] + weights[1:] * samples[1:]) / (weights[:-1] + weights[1:])
        
        xp = np.empty(2 * len(midpoints) -1)
        fp = np.empty_like(xp)
        
        xp[::2] = midpoints
        fp[::2] = samples
        xp[1::2] = edges[1:-1]
        fp[1::2] = sample_midpoints
        
        return np.interp(q, xp, fp)
        
