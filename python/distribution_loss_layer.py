import sys
import yaml

import caffe
import numpy as np
from scipy.special import gamma
from scipy.special import gammaln
from scipy.special import polygamma
from scipy.stats import beta


# assign points to grid bins
def getPlaces(x, grid):
    places_to_bins = dict()  # i of sorted x to j in grid
    bins_to_places = dict()

    for i in xrange(len(grid)):
        bins_to_places[i] = list()
    inx_sorted = np.argsort(x)

    ind = 1
    # find initial bucket :
    for i in xrange(len(grid)):
        if x[inx_sorted[0]] > grid[i]:
            ind = i + 1
        else:
            break

    x_start = 0
    while x[inx_sorted[x_start]] < grid[0]:
        x_start += 1

    for i in xrange(x_start, len(x)):
        while x[inx_sorted[i]] > grid[ind]:
            ind += 1
            if ind >= len(grid):
                return places_to_bins, bins_to_places
        places_to_bins[inx_sorted[i]] = ind
        bins_to_places[ind].append(inx_sorted[i])
    return places_to_bins, bins_to_places


# estimate the histogram using the assigments of points to grid bins
def getDistributionDensity(x, bins_to_places, grid, grid_delta):
    p = np.zeros_like(grid)
    for i in xrange(len(grid)):

        left_add = 0
        if i > 0:
            d_i_list_left = np.array(bins_to_places[i])
            left_dist = np.array([x[ii] for ii in d_i_list_left])
            left_add = sum(left_dist - grid[i - 1])

        right_add = 0
        if i < len(grid) - 1:
            d_i_list_right = np.array(bins_to_places[i + 1])
            right_dist = np.array([x[ii] for ii in d_i_list_right])
            right_add = sum(grid[i + 1] - right_dist)

        p[i] = (left_add + right_add)

    p /= len(x) * grid_delta
    return p


# def calculateNPGradOverBins(d_pos, distr_pos, d_neg, distr_neg, grid_delta):
#    dldp = np.cumsum(distr_neg[::-1])[::-1]
#    dldn = np.cumsum(distr_pos)
#    
#    grad_pos = dldp[:]
#    grad_pos[1:] = (grad_pos[1:] - grad_pos[:-1])
#    grad_pos /= grid_delta*len(d_pos)
#    
#    grad_neg = dldn[:]
#    grad_neg[1:] = (grad_neg[1:] - grad_neg[:-1])
#    grad_neg/= grid_delta*len(d_neg)
#    return grad_pos, grad_neg

def calculateLossGradOverDistribution(distr_pos, distr_neg, L):
    grad_pos = np.dot(L, distr_neg)
    grad_neg = np.dot(distr_pos, L)
    return grad_pos, grad_neg


def calculateLossGradOverBinsForHist(d_pos, d_neg, grid_delta, grad_pos, grad_neg):
    grad_pos[1:] = (grad_pos[1:] - grad_pos[:-1])
    grad_pos /= grid_delta * len(d_pos)

    grad_neg[1:] = (grad_neg[1:] - grad_neg[:-1])
    grad_neg /= grid_delta * len(d_neg)

    return grad_pos, grad_neg


def getGradOverData(data, grad_over_bins, places_to_bins):
    grad = []
    for i in xrange(len(data)):
        grad.append(grad_over_bins[places_to_bins[i]])

    return np.array(grad)


##################### Beta-distribution fitting and gradient ##########################################################
# estimate beta-distribution
def getBetaDistributionDensity(x, grid, grid_delta):
    grid = np.array(np.copy(grid))
    x = np.array([x[i] for i in xrange(len(x)) if x[i] >= -1 and x[i] <= 1])

    x_scaled = (x + 1.) / 2.

    mean = np.mean(x_scaled)
    var = np.var(x_scaled, ddof=1)
    alpha1 = mean ** 2 * (1 - mean) / var - mean
    beta1 = alpha1 * (1 - mean) / mean

    fitted = lambda x, a, b: gamma(a + b) / gamma(a) / gamma(b) * x ** (a - 1) * (1 - x) ** (b - 1)  # pdf of beta

    grid_scaled = np.array((grid + 1) / 2)

    ### to avoid  zero devision errors
    grid_scaled[0] = 1e-5
    grid_scaled[len(grid_scaled) - 1] = 0.999

    distr_ = beta.pdf(grid_scaled, alpha1, beta1) * grid_delta / (2.)

    return distr_


def gamma_derivative(x):
    return polygamma(0, x) * gamma(x)


def dvardx(x):
    meanx_ = np.mean(x)
    expr1 = (x - meanx_) * (-1) * 2.0 / (len(x) - 1) / len(x)
    expr3 = np.ones((1, len(x))) * np.sum(expr1) * 2.0 / (len(x) - 1) / len(x)
    expr4 = (x - meanx_) * 2. / (len(x) - 1)
    dvardx = expr3 + expr4
    return dvardx


def calculateLossGradOverDataForBeta(d_pos, d_neg, grid, grid_delta, grad_pos, grad_neg):
    grid = np.array(np.copy(grid))
    # scale grid
    grid = np.array((grid + 1.) / 2.)
    ### to avoid  zero devision errors
    grid[0] = 1e-5
    grid[len(grid) - 1] = 0.999

    d_pos[d_pos >= 1] = 1
    d_pos[d_pos <= -1] = -1

    d_pos_scaled = (d_pos + 1.) / 2.
    mean_pos = np.mean(d_pos_scaled)
    var_pos = np.var(d_pos_scaled, ddof=1)
    alpha_pos = mean_pos ** 2 * (1 - mean_pos) / var_pos - mean_pos
    beta_pos = alpha_pos * (1 - mean_pos) / mean_pos

    d_neg[d_neg >= 1] = 1
    d_neg[d_neg <= -1] = -1
    d_neg_scaled = (d_neg + 1.) / 2.
    mean_neg = np.mean(d_neg_scaled)
    var_neg = np.var(d_neg_scaled, ddof=1)
    alpha_neg = mean_neg ** 2 * (1 - mean_neg) / var_neg - mean_neg
    beta_neg = alpha_neg * (1 - mean_neg) / mean_neg

    # dLd_distr - checked
    dldp = grad_pos
    dldn = grad_neg

    # dmeandx - checked
    dmean_posdd_pos = np.ones((1, len(d_pos))) * 1.0 / len(d_pos)
    dmean_negdd_neg = np.ones((1, len(d_neg))) * 1.0 / len(d_neg)

    # dvardx - checked
    dvar_posdd_pos = dvardx(d_pos_scaled)
    dvar_negdd_neg = dvardx(d_neg_scaled)

    ######## d alpha/beta d mean/var

    # checked
    dalpha_dmean_pos = 1. / var_pos * (2 * mean_pos - 3 * mean_pos ** 2) - 1 + \
                       mean_pos ** 2 * (1 - mean_pos) / var_pos ** 2 / (len(d_pos) - 1) * (
                           2 * np.sum(d_pos_scaled - mean_pos))
    dalpha_dmean_neg = 1. / var_neg * (2 * mean_neg - 3 * mean_neg ** 2) - 1 + \
                       mean_neg ** 2 * (1 - mean_neg) / var_neg ** 2 / (len(d_neg) - 1) * (
                           2 * np.sum(d_neg_scaled - mean_neg))

    # checked
    dalpha_dvar_pos = -(mean_pos) ** 2 * (1 - mean_pos) * (var_pos) ** (-2)
    dalpha_dvar_neg = -(mean_neg) ** 2 * (1 - mean_neg) * (var_neg) ** (-2)

    # checked
    dbeta_dmean_pos = -alpha_pos / (mean_pos) ** 2 + (1 - mean_pos) / mean_pos * dalpha_dmean_pos
    dbeta_dmean_neg = -alpha_neg / (mean_neg) ** 2 + (1 - mean_neg) / mean_neg * dalpha_dmean_neg

    # checked
    dbeta_dvar_pos = (1 - mean_pos) / mean_pos * dalpha_dvar_pos
    dbeta_dvar_neg = (1 - mean_neg) / mean_neg * dalpha_dvar_neg

    ###### d aplha/beta d x - checheked
    dalpha_dd_pos = dalpha_dmean_pos * dmean_posdd_pos + dalpha_dvar_pos * dvar_posdd_pos
    dalpha_dd_neg = dalpha_dmean_neg * dmean_negdd_neg + dalpha_dvar_neg * dvar_negdd_neg

    dbeta_dd_pos = dbeta_dmean_pos * dmean_posdd_pos + dbeta_dvar_pos * dvar_posdd_pos
    dbeta_dd_neg = dbeta_dmean_neg * dmean_negdd_neg + dbeta_dvar_neg * dvar_negdd_neg

    ### d distr(p/n) d alpha/beta


    gammaTerm_pos = np.exp(gammaln(alpha_pos + beta_pos) - gammaln(alpha_pos) - \
                           gammaln(beta_pos))
    gammaTerm_neg = np.exp(gammaln(alpha_neg + beta_neg) - gammaln(alpha_neg) - \
                           gammaln(beta_neg))

    # checked
    dGammaTerm_dalpha_pos = gammaTerm_pos * (polygamma(0, alpha_pos + beta_pos) - polygamma(0, alpha_pos))
    dGammaTerm_dalpha_neg = gammaTerm_neg * (polygamma(0, alpha_neg + beta_neg) - polygamma(0, alpha_neg))

    # checked
    dGammaTerm_dbeta_pos = gammaTerm_pos * (polygamma(0, alpha_pos + beta_pos) - polygamma(0, beta_pos))
    dGammaTerm_dbeta_neg = gammaTerm_neg * (polygamma(0, alpha_neg + beta_neg) - polygamma(0, beta_neg))

    dpdalpha_pos = (dGammaTerm_dalpha_pos * grid ** (alpha_pos - 1) * (1 - grid) ** (beta_pos - 1) +
                    gammaTerm_pos * grid ** (alpha_pos - 1) * np.log(grid) * (1 - grid) ** (
                        beta_pos - 1)) * grid_delta / 2.
    dndalpha_neg = (dGammaTerm_dalpha_neg * grid ** (alpha_neg - 1) * (1 - grid) ** (beta_neg - 1) +
                    gammaTerm_neg * grid ** (alpha_neg - 1) * np.log(grid) * (1 - grid) ** (
                        beta_neg - 1)) * grid_delta / 2.

    dpdbeta_pos = (dGammaTerm_dbeta_pos * grid ** (alpha_pos - 1) * (1 - grid) ** (beta_pos - 1) +
                   gammaTerm_pos * grid ** (alpha_pos - 1) * (1 - grid) ** (beta_pos - 1) * np.log(
                       1 - grid)) * grid_delta / 2.
    dndbeta_neg = (dGammaTerm_dbeta_neg * grid ** (alpha_neg - 1) * (1 - grid) ** (beta_neg - 1) +
                   gammaTerm_neg * grid ** (alpha_neg - 1) * (1 - grid) ** (beta_neg - 1) * np.log(
                       1 - grid)) * grid_delta / 2.

    # d distr d x
    # matrix : grid X number of points

    dpdd_pos = np.dot(dpdalpha_pos.T.reshape((len(grid), 1)), dalpha_dd_pos) + \
               np.dot(dpdbeta_pos.T.reshape((len(grid), 1)), dbeta_dd_pos)
    dndd_neg = np.dot(dndalpha_neg.T.reshape((len(grid), 1)), dalpha_dd_neg) + \
               np.dot(dndbeta_neg.T.reshape((len(grid), 1)), dbeta_dd_neg)

    ############# FINAL GRADIENT
    grad_pos = np.dot(dldp.reshape((1, len(grid))), dpdd_pos)
    grad_neg = np.dot(dldn.reshape((1, len(grid))), dndd_neg)

    # need scaling as beta distribution is fitted on scaled data
    return np.array(grad_pos / 2.).reshape(len(d_pos)), np.array(grad_neg / 2.).reshape(len(d_neg))


#######################################################################################################################
LOSS_SIMPLE = 'simple'
LOSS_LINEAR = 'linear'
LOSS_EXP = 'exp'

DISTR_TYPE_HIST = 'hist'
DISTR_TYPE_BETA = 'beta'


# Calculates probability of wrong order in pairs' similarities: positive pair less similar than negative one
# (this corresponds to 'simple' loss, other variants ('linear', 'exp') are generalizations that take into account 
# not only the order but also the difference between the two similarity values).
# Can use histogram and beta-distribution to fit input data.
class DistributionLossLayer(caffe.Layer):
    def getL(self):
        L = np.ones((len(self.grid), len(self.grid)))
        if self.loss == LOSS_SIMPLE:
            for i in xrange(len(self.grid)):
                L[i] = self.grid[i] <= self.grid
        elif self.loss == LOSS_LINEAR:
            for i in xrange(len(self.grid)):
                L[i] = self.margin - self.grid[i] + self.grid
            L[L < 0] = 0
        elif self.loss == LOSS_EXP:
            for i in xrange(len(self.grid)):
                L[i] = np.log(np.exp(self.alpha * (self.margin + self.grid - self.grid[i])) + 1)
        return L

    def setup(self, bottom, top):
        # np.seterr(all='raise')
        layer_params = yaml.load(self.param_str)
        print layer_params
        sys.stdout.flush()

        self.iteration = 0
        # parameters for the Histogram loss generalization variants
        self.alpha = 1
        if 'alpha' in layer_params:
            self.alpha = layer_params['alpha']
        self.margin = 0
        if 'margin' in layer_params:
            self.margin = layer_params['margin']

        # loss type
        self.loss = LOSS_SIMPLE
        if 'loss' in layer_params:
            self.loss = layer_params['loss']
        if self.loss not in [LOSS_SIMPLE, LOSS_LINEAR, LOSS_EXP]:
            raise Exception('unknown loss : ' + self.loss)

        self.distr_type = DISTR_TYPE_HIST
        if 'distr_type' in layer_params:
            self.distr_type = layer_params['distr_type']
        if self.distr_type not in [DISTR_TYPE_HIST, DISTR_TYPE_BETA]:
            raise Exception('unknown distribution : ' + self.distr_type)

        self.grid_delta = layer_params['grid_delta']
        self.grid = np.array([i for i in np.arange(-1., 1. + self.grid_delta, self.grid_delta)])
        self.pos_label = 1
        self.neg_label = -1

    def reshape(self, bottom, top):
        ## bottom[0] is cosine similarities
        ## bottom[1] is pair labels

        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension: " + str(bottom[0].count) + " " + str(bottom[1].count))
        if not bottom[0].channels == bottom[0].height == bottom[0].width:
            raise Exception("Similirities are not scalars.")
        if not bottom[1].channels == bottom[1].height == bottom[1].width:
            raise Exception("Pair labels are not scalars.")
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.d_pos = []
        self.d_neg = []

        bottom[0].data[bottom[0].data >= 1.] = 1.
        bottom[0].data[bottom[0].data <= -1.] = -1.

        self.pos_indecies = bottom[1].data == self.pos_label
        self.neg_indecies = bottom[1].data == self.neg_label
        self.d_pos = bottom[0].data[self.pos_indecies]
        self.d_neg = bottom[0].data[self.neg_indecies]

        self.d_pos = np.array(self.d_pos)
        self.d_neg = np.array(self.d_neg)

        self.places_to_bins_pos, self.bins_to_places_pos = getPlaces(self.d_pos, self.grid)
        self.places_to_bins_neg, self.bins_to_places_neg = getPlaces(self.d_neg, self.grid)

        if self.distr_type == DISTR_TYPE_HIST:
            self.distr_pos = getDistributionDensity(self.d_pos, self.bins_to_places_pos, self.grid, self.grid_delta)
            self.distr_neg = getDistributionDensity(self.d_neg, self.bins_to_places_neg, self.grid, self.grid_delta)

        if self.distr_type == DISTR_TYPE_BETA:
            self.distr_pos = getBetaDistributionDensity(self.d_pos, self.grid, self.grid_delta)
            self.distr_neg = getBetaDistributionDensity(self.d_neg, self.grid, self.grid_delta)

        L = self.getL()
        top[0].data[...] = np.dot(np.dot(self.distr_pos, L), self.distr_neg)

        sys.stdout.flush()
        self.iteration += 1

    def backward(self, top, propagate_down, bottom):
        L = self.getL()
        grad_pos_distr, grad_neg_distr = calculateLossGradOverDistribution(self.distr_pos, self.distr_neg, L)

        if self.distr_type == DISTR_TYPE_HIST:
            self.grad_pos_bin, self.grad_neg_bin = calculateLossGradOverBinsForHist(self.d_pos, self.d_neg,
                                                                                    self.grid_delta, grad_pos_distr,
                                                                                    grad_neg_distr)
            self.grad_pos = getGradOverData(self.d_pos, self.grad_pos_bin, self.places_to_bins_pos)
            self.grad_neg = getGradOverData(self.d_neg, self.grad_neg_bin, self.places_to_bins_neg)
        elif self.distr_type == DISTR_TYPE_BETA:
            self.grad_pos, self.grad_neg = calculateLossGradOverDataForBeta(self.d_pos, self.d_neg, self.grid,
                                                                            self.grid_delta, grad_pos_distr,
                                                                            grad_neg_distr)

        grad = np.zeros((len(self.grad_pos) + len(self.grad_neg), 1, 1, 1))
        grad[self.pos_indecies] = self.grad_pos
        grad[self.neg_indecies] = self.grad_neg
        bottom[0].diff[...] = grad
