from ANNarchy import (
    Population,
    Projection,
    get_projection,
    Network,
    get_population,
    Monitor,
    raster_plot,
    setup,
    clear,
    populations,
    dt,
    SpikeSourceArray,
)
from ANNarchy.core.Global import _network
from CompNeuroPy.neuron_models import (
    poisson_neuron_up_down,
    Izhikevich2003_flexible_noisy_I,
)
from CompNeuroPy import generate_model, cnp_clear, rmse
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from time import time

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


#######   FUNCTIONS   ######
def train_regr(X_raw, y_raw, X_name_list, y_name):
    """
    shape X = (n_samples, n_features), y = (n_samples, 1)
    """
    scaler_X = preprocessing.StandardScaler().fit(X_raw)
    scaler_y = preprocessing.StandardScaler().fit(y_raw)

    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw)[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.1, random_state=42
    )

    ### Regression
    regr_model = MLPRegressor(
        hidden_layer_sizes=(300, 200, 100), random_state=42, max_iter=100
    )
    regr_model.fit(X_train, y_train)

    ### test predictions
    pred_test_scaled = regr_model.predict(X_test)
    test_error_scaled = rmse(y_test, pred_test_scaled)

    y_test_raw = scaler_y.inverse_transform(y_test[:, None])
    pred_test_raw = scaler_y.inverse_transform(pred_test_scaled[:, None])
    test_error_raw = rmse(y_test_raw, pred_test_raw)

    plt.figure()
    plt.subplot(211)
    plt.title("scaled")
    sort_idx_arr = np.argsort(y_test)
    plt.plot(y_test[sort_idx_arr], label="target")
    plt.plot(pred_test_scaled[sort_idx_arr], label="pred", alpha=0.3)
    plt.legend()
    plt.subplot(212)
    plt.title("raw")
    plt.plot(y_test_raw[sort_idx_arr], label="target")
    plt.plot(pred_test_raw[sort_idx_arr], label="pred", alpha=0.3)
    plt.ylim([-1, 10])
    plt.legend()
    plt.tight_layout()
    plt.savefig("tmp_pred.png")
    plt.close("all")
    print("test_error scaled:", test_error_scaled)
    print("test_error raw:", test_error_raw)

    return regr_function_3p(regr_model, scaler_X, scaler_y, X_name_list, y_name)


class regr_function_3p:
    def __init__(self, regr_model, scaler_X, scaler_y, X_name_list, y_name) -> None:
        self.regr_model = regr_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        self.X_name_dict = {
            ["p1", "p2", "p3"][idx_X_name]: X_name
            for idx_X_name, X_name in enumerate(X_name_list)
        }
        self.y_name = y_name

    def __call__(self, p1=None, p2=None, p3=None):
        """
        params:
            p1, number or array, default 0
            p2, number or array, default 0
            p3, number or array, default 0

            if p1, p2, p3 == array --> same size!

        return:
            y_arr
        """
        ### check which values are given
        p1_given = not (isinstance(p1, type(None)))
        p2_given = not (isinstance(p2, type(None)))
        p3_given = not (isinstance(p3, type(None)))

        ### reshape all to target shape
        p1 = np.array(p1).reshape((-1, 1)).astype(float)
        p2 = np.array(p2).reshape((-1, 1)).astype(float)
        p3 = np.array(p3).reshape((-1, 1)).astype(float)

        ### check if arrays of given values have same size
        size_arr = np.array([p1.shape[0], p2.shape[0], p3.shape[0]])
        given_arr = np.array([p1_given, p2_given, p3_given]).astype(bool)
        if len(size_arr[given_arr]) > 0:
            all_size = size_arr[given_arr][0]
            all_same_size = np.all(size_arr[given_arr] == all_size)
            if not all_same_size:
                raise ValueError(
                    "regr_function_3p call: given p1, p2 and p3 have to have same size!"
                )

        ### set the correctly sized arrays for the not given values
        if not p1_given:
            p1 = np.zeros(all_size).reshape((-1, 1))
        if not p2_given:
            p2 = np.zeros(all_size).reshape((-1, 1))
        if not p3_given:
            p3 = np.zeros(all_size).reshape((-1, 1))

        ### predict y_arr
        X_raw = np.concatenate([p1, p2, p3], axis=1)
        y_arr = pred_regr(X_raw, self.regr_model, self.scaler_X, self.scaler_y)[:, 0]

        return y_arr


def pred_regr(X_raw, regr_model, scaler_X, scaler_y):
    """
    X shape = (n_samples, n_features)

    pred shape = (n_samples, 1)
    """

    X_scaled = scaler_X.transform(X_raw)
    pred_scaled = regr_model.predict(X_scaled)
    pred_raw = scaler_y.inverse_transform(pred_scaled[:, None])

    return pred_raw


def get_rate_1000(
    net, population, variable_init_dict, monitor, I_app=0, g_ampa=0, g_gaba=0
):
    """
    simulates 1000 ms a population and returns the firing rate of each neuron
    """
    ### reset and set init values
    net.reset()
    for var_name, var_val in variable_init_dict.items():
        tmp = getattr(population, var_name)
        set_val = np.ones(len(tmp)) * var_val
        setattr(population, var_name, set_val)
    ### slow down conductances (i.e. make them constant)
    population.tau_ampa = 1e20
    population.tau_gaba = 1e20
    ### simulate
    population.I_app = I_app
    population.g_ampa = g_ampa
    population.g_gaba = g_gaba
    net.simulate(1000)
    ### rate is nbr spikes since simulated 1000 ms
    spike_dict = monitor.get("spike")
    f_arr = np.zeros(len(population))
    for idx_n, n in enumerate(spike_dict.keys()):
        rate = len(spike_dict[n])
        f_arr[idx_n] = rate

    return f_arr


def get_g_1000(net_dict, afferent_projection_dict):
    """
    simulates 1000 ms a population consisting of a single neuron and returns the g_ampa and g_gaba value
    """
    net = net_dict["net"]
    monitor = net_dict["monitor"]
    ampa_inp_proj = net_dict["ampa_inp_proj"]
    gaba_inp_proj = net_dict["gaba_inp_proj"]
    ### reset
    net.reset()
    ### set weights of the projections
    ampa_weight_list = []
    if not isinstance(ampa_inp_proj, type(None)):
        for idx_proj in range(len(afferent_projection_dict["projection_names"])):
            if afferent_projection_dict["target"][idx_proj] == "ampa":
                ampa_weight_list.append(afferent_projection_dict["weights"][idx_proj])
        ampa_inp_proj.w = np.array(ampa_weight_list)
    gaba_weight_list = []
    if not isinstance(gaba_inp_proj, type(None)):
        for idx_proj in range(len(afferent_projection_dict["projection_names"])):
            if afferent_projection_dict["target"][idx_proj] == "gaba":
                gaba_weight_list.append(afferent_projection_dict["weights"][idx_proj])
        gaba_inp_proj.w = np.array(gaba_weight_list)
    ### simulate
    net.simulate(1000)
    ### get g values
    g_ampa_arr = monitor.get("g_ampa")[:, 0]
    g_gaba_arr = monitor.get("g_gaba")[:, 0]
    g_ampa_val = np.mean(g_ampa_arr[int(len(g_ampa_arr) * 0.7) :])
    g_gaba_val = np.mean(g_gaba_arr[int(len(g_gaba_arr) * 0.7) :])

    return [g_ampa_val, g_gaba_val]


def predict_1d(X, y, X_pred):
    y_X = interp1d(x=X, y=y, fill_value="extrapolate")
    y_pred = float(y_X(X_pred))
    return y_pred


def get_init_neuron_variables(net, population):
    net.reset()
    population.I_app = 0
    net.simulate(2000)
    variable_init_dict = {
        var_name: getattr(population, var_name) for var_name in population.variables
    }
    net.reset()
    return variable_init_dict


def prepare_f_I_g_curve(net_single_dict, f_t, population_name):

    net = net_single_dict["net"]
    population = net_single_dict["population"]
    monitor = net_single_dict["monitor"]
    variable_init_dict = net_single_dict["variable_init_dict"]

    ### get f_0
    f_0 = get_rate_1000(net, population, variable_init_dict, monitor)[0]
    f_max = f_0 + f_t + 100

    ### now use different I_app values to get a f(I) curve
    ### if f(0)>0 also explor negative I values
    ### first check I_app=3
    ### rate should increase
    ### expect linear increase and select new I_app to get f between f_max and f_max
    ### repeat this until I_app for f_max found --> I_max
    ### than bounds for I for f(I) curver are -I_max and I_max
    init_I_app = 3
    f_rec = f_0
    alpha_tol = 0.001
    tolerance = (f_max - f_0) * alpha_tol
    n_it_max = 100
    n_it = 0
    I_app = init_I_app
    I_list = [0]
    f_list = [f_0]
    print(population_name)
    print(f"f_max: {f_max}")
    while (
        not (0 <= (f_rec - f_max) and (f_rec - f_max) < tolerance) and n_it < n_it_max
    ):
        ### get f for I
        f_rec = get_rate_1000(
            net, population, variable_init_dict, monitor, I_app=I_app
        )[0]
        ### append I_app and f_rec to f_list/I_list
        f_list.append(f_rec)
        I_list.append(I_app)
        ### predict new I_app for f_max
        I_app = predict_1d(X=f_list, y=I_list, X_pred=f_max)
        ### increase iterator
        n_it += 1

    if not (n_it < n_it_max):
        raise UserWarning(
            f"could not find I_max for f_max={f_max} for population {population_name}"
        )
    I_max = I_list[-1]
    print(np.array([I_list, f_list]).T)

    ### now get g_ampa_max, by increasing g_ampa until f_max is reached (same as with I_app)
    init_g_ampa = 1
    f_rec = f_0
    alpha_tol = 0.001
    tolerance = (f_max - f_0) * alpha_tol
    n_it_max = 100
    n_it = 0
    g_ampa = init_g_ampa
    g_ampa_list = [0]
    f_list = [f_0]
    while (
        not (0 <= (f_rec - f_max) and (f_rec - f_max) < tolerance) and n_it < n_it_max
    ):
        ### get f for g_ampa
        f_rec = get_rate_1000(
            net, population, variable_init_dict, monitor, g_ampa=g_ampa
        )[0]
        ### append I_app and f_rec to f_list/I_list
        f_list.append(f_rec)
        g_ampa_list.append(g_ampa)
        ### predict new g_ampa_list for f_max
        g_ampa = predict_1d(X=f_list, y=g_ampa_list, X_pred=f_max)
        ### increase iterator
        n_it += 1

    if not (n_it < n_it_max):
        raise UserWarning(
            f"could not find g_ampa_max for f_max={f_max} for population {population_name}"
        )
    g_ampa_max = g_ampa_list[-1]
    print(np.array([g_ampa_list, f_list]).T)

    ### now get g_gaba_max, setting I_app to I_max and increasing g_gaba until f = f_0
    ### now get g_ampa_max, by increasing g_ampa until f_max is reached (same as with I_app)
    init_g_gaba = 1
    f_rec = f_max
    alpha_tol = 0.001
    tolerance = (f_0 - f_max) * alpha_tol
    n_it_max = 100
    n_it = 0
    g_gaba = init_g_gaba
    g_gaba_list = [0]
    f_list = [f_max]
    print(f"f_0: {f_0}")
    while not (0 >= (f_rec - f_0) and (f_rec - f_0) > tolerance) and n_it < n_it_max:
        ### get f for g_gaba
        f_rec = get_rate_1000(
            net, population, variable_init_dict, monitor, g_gaba=g_gaba, I_app=I_max
        )[0]
        ### append I_app and f_rec to f_list/I_list
        f_list.append(f_rec)
        g_gaba_list.append(g_gaba)
        ### predict new g_gaba_list for f_0
        g_gaba = predict_1d(X=f_list, y=g_gaba_list, X_pred=f_0)
        ### increase iterator
        n_it += 1

    if not (n_it < n_it_max):
        raise UserWarning(
            f"could not find g_gaba_max for f_0={f_0} for population {population_name}"
        )
    g_gaba_max = g_gaba_list[-1]
    print(np.array([g_gaba_list, f_list]).T)

    return [I_max, g_ampa_max, g_gaba_max, variable_init_dict]


def get_f_I_g_curve(net_many_dict, prepare_list):

    I_max, g_ampa_max, g_gaba_max, variable_init_dict = prepare_list

    ### now fill f(I,g_ampa,g_gaba) curve between I, g_ampa and g_gaba bounds
    number_of_points = np.round(len(net_many_dict["population"]) ** (1 / 3), 0).astype(
        int
    )
    I_app_value_array = np.linspace(-I_max, I_max, number_of_points)
    g_ampa_value_array = np.linspace(0, g_ampa_max, number_of_points)
    g_gaba_value_array = np.linspace(0, g_gaba_max, number_of_points)

    ### get all combinations of I, g_ampa and g_gaba
    I_g_arr = np.array(
        list(
            itertools.product(
                *[I_app_value_array, g_ampa_value_array, g_gaba_value_array]
            )
        )
    )
    I_app_arr = I_g_arr[:, 0]
    g_ampa_arr = I_g_arr[:, 1]
    g_gaba_arr = I_g_arr[:, 2]

    ### get f for all combinations of I, g_ampa and g_gaba
    f_rec_arr = get_rate_1000(
        net_many_dict["net"],
        net_many_dict["population"],
        variable_init_dict,
        net_many_dict["monitor"],
        g_ampa=g_ampa_arr,
        g_gaba=g_gaba_arr,
        I_app=I_app_arr,
    )
    ### add f_rec to tje f_I_g_data_arr
    f_I_g_data_arr = np.concatenate([I_g_arr, f_rec_arr[:, None]], axis=1)

    print("train regr...")
    f_I_g_curve = train_regr(
        X_raw=f_I_g_data_arr[:, :3],
        y_raw=f_I_g_data_arr[:, 3][:, None],
        X_name_list=["I", "g_ampa", "g_gaba"],
        y_name="f",
    )
    I_f_g_curve = train_regr(
        X_raw=f_I_g_data_arr[:, 1:],
        y_raw=f_I_g_data_arr[:, 0][:, None],
        X_name_list=["g_ampa", "g_gaba", "f"],
        y_name="I",
    )

    return [f_I_g_curve, I_f_g_curve, I_max, g_ampa_max]


def BGM_part_function(params):
    #######   POPULATIONS   ######
    ### cortex / input populations
    cor = Population(
        params["cor.size"],
        poisson_neuron_up_down,
        name="cor",
    )
    cor.tau_up = params["cor.tau_up"]
    cor.tau_down = params["cor.tau_down"]
    ### BG Populations
    stn = Population(
        params["stn.size"],
        Izhikevich2003_flexible_noisy_I,
        name="stn",
    )
    stn.a = params["stn.a"]
    stn.b = params["stn.b"]
    stn.c = params["stn.c"]
    stn.d = params["stn.d"]
    stn.n2 = params["stn.n2"]
    stn.n1 = params["stn.n1"]
    stn.n0 = params["stn.n0"]
    stn.tau_ampa = params["stn.tau_ampa"]
    stn.tau_gaba = params["stn.tau_gaba"]
    stn.E_ampa = params["stn.E_ampa"]
    stn.E_gaba = params["stn.E_gaba"]
    snr = Population(
        params["snr.size"],
        Izhikevich2003_flexible_noisy_I,
        name="snr",
    )
    snr.a = params["snr.a"]
    snr.b = params["snr.b"]
    snr.c = params["snr.c"]
    snr.d = params["snr.d"]
    snr.n2 = params["snr.n2"]
    snr.n1 = params["snr.n1"]
    snr.n0 = params["snr.n0"]
    snr.tau_ampa = params["snr.tau_ampa"]
    snr.tau_gaba = params["snr.tau_gaba"]
    snr.E_ampa = params["snr.E_ampa"]
    snr.E_gaba = params["snr.E_gaba"]
    gpe = Population(
        params["gpe.size"],
        Izhikevich2003_flexible_noisy_I,
        name="gpe",
    )
    gpe.a = params["gpe.a"]
    gpe.b = params["gpe.b"]
    gpe.c = params["gpe.c"]
    gpe.d = params["gpe.d"]
    gpe.n2 = params["gpe.n2"]
    gpe.n1 = params["gpe.n1"]
    gpe.n0 = params["gpe.n0"]
    gpe.tau_ampa = params["gpe.tau_ampa"]
    gpe.tau_gaba = params["gpe.tau_gaba"]
    gpe.E_ampa = params["gpe.E_ampa"]
    gpe.E_gaba = params["gpe.E_gaba"]
    thal = Population(
        params["thal.size"],
        Izhikevich2003_flexible_noisy_I,
        name="thal",
    )
    thal.a = params["thal.a"]
    thal.b = params["thal.b"]
    thal.c = params["thal.c"]
    thal.d = params["thal.d"]
    thal.n2 = params["thal.n2"]
    thal.n1 = params["thal.n1"]
    thal.n0 = params["thal.n0"]
    thal.tau_ampa = params["thal.tau_ampa"]
    thal.tau_gaba = params["thal.tau_gaba"]
    thal.E_ampa = params["thal.E_ampa"]
    thal.E_gaba = params["thal.E_gaba"]

    ######   PROJECTIONS   ######
    ### cortex go output
    cor__stn = Projection(
        pre=cor,
        post=stn,
        target="ampa",
        name="cor__stn",
    )
    cor__stn.connect_fixed_probability(
        probability=params["cor__stn.probability"], weights=1
    )
    ### stn output
    stn__snr = Projection(
        pre=stn,
        post=snr,
        target="ampa",
        name="stn__snr",
    )
    stn__snr.connect_fixed_probability(
        probability=params["stn__snr.probability"], weights=1
    )
    stn__gpe = Projection(
        pre=stn,
        post=gpe,
        target="ampa",
        name="stn__gpe",
    )
    stn__gpe.connect_fixed_probability(
        probability=params["stn__gpe.probability"], weights=1
    )
    ### gpe proto output
    gpe__stn = Projection(
        pre=gpe,
        post=stn,
        target="gaba",
        name="gpe__stn",
    )
    gpe__stn.connect_fixed_probability(
        probability=params["gpe__stn.probability"], weights=1
    )
    gpe__snr = Projection(
        pre=gpe,
        post=snr,
        target="gaba",
        name="gpe__snr",
    )
    gpe__snr.connect_fixed_probability(
        probability=params["gpe__snr.probability"], weights=1
    )
    ### snr output
    snr__thal = Projection(
        pre=snr,
        post=thal,
        target="gaba",
        name="snr__thal",
    )
    snr__thal.connect_fixed_probability(
        probability=params["snr__thal.probability"], weights=1
    )
    snr__snr = Projection(
        pre=snr,
        post=snr,
        target="gaba",
        name="snr__snr",
    )
    snr__snr.connect_fixed_probability(
        probability=params["snr__snr.probability"], weights=1
    )


def get_duration_1000(n_neuron, model, population_name):
    cnp_clear()
    model.create(do_compile=False)
    ### create many neuron population
    many_neuron = Population(
        n_neuron,
        neuron=get_population(population_name).neuron_type,
        name="many_neuron",
    )
    for attr_name, attr_val in get_population(population_name).init.items():
        setattr(many_neuron, attr_name, attr_val)
    ### create Monitor for many neuron population
    mon_many = Monitor(many_neuron, ["spike", "g_ampa", "g_gaba"])

    net = Network()
    net.add([many_neuron, mon_many])
    net.compile()
    start = time()
    net.simulate(1000)
    end = time()
    duration = end - start

    return duration


def get_nr_many_neurons(model, population_name):
    duration_1000 = get_duration_1000(1000, model, population_name)
    nr_factor = 10 / duration_1000
    nr_neuron_10s = int(int(int(1000 * nr_factor) ** (1 / 3)) ** 3)
    return nr_neuron_10s


def create_single_network(population_name):
    ### create single neuron from population
    single_neuron = Population(
        1, neuron=get_population(population_name).neuron_type, name="single_neuron"
    )
    for attr_name, attr_val in get_population(population_name).init.items():
        setattr(single_neuron, attr_name, attr_val)

    ### create Monitor for single neuron
    mon_single = Monitor(single_neuron, ["spike"])

    ### create network with single neuron
    net_single = Network()
    net_single.add([single_neuron, mon_single])
    net_single.compile()

    variable_init_dict = get_init_neuron_variables(
        net_single, net_single.get(single_neuron)
    )

    ### network dict
    net_single_dict = {
        "net": net_single,
        "population": net_single.get(single_neuron),
        "monitor": net_single.get(mon_single),
        "variable_init_dict": variable_init_dict,
    }
    return net_single_dict


def create_many_network(model, population_name):
    ### create many neuron population whose simulation of 1000ms takes about 10sec
    nr_many_neurons = get_nr_many_neurons(model, population_name)
    print("nr_many_neurons:", nr_many_neurons)
    many_neuron = Population(
        nr_many_neurons,
        neuron=get_population(population_name).neuron_type,
        name="many_neuron",
    )
    for attr_name, attr_val in get_population(population_name).init.items():
        setattr(many_neuron, attr_name, attr_val)

    ### create Monitor for many neuron
    mon_many = Monitor(many_neuron, ["spike"])

    ### create network with many neuron
    net_many = Network()
    net_many.add([many_neuron, mon_many])
    net_many.compile()

    ### network dict
    net_many_dict = {
        "net": net_many,
        "population": net_many.get(many_neuron),
        "monitor": net_many.get(mon_many),
    }

    return net_many_dict


def get_afferent_projection_dict(model, population_name):
    afferent_projection_dict = {}
    afferent_projection_dict["projection_names"] = []
    for projection in model.projections:
        if get_projection(projection).post.name == population_name:
            afferent_projection_dict["projection_names"].append(projection)

    ### get target firing rates resting-state for afferent projections
    afferent_projection_dict["target firing rate"] = []
    afferent_projection_dict["probability"] = []
    afferent_projection_dict["size"] = []
    afferent_projection_dict["target"] = []
    for projection in afferent_projection_dict["projection_names"]:
        pre_pop_name = get_projection(projection).pre.name
        ### target firing rate
        afferent_projection_dict["target firing rate"].append(
            target_firing_rate_dict[pre_pop_name]
        )
        ### probability, _connection_args only if connect_fixed_prob (i.e. connector_name==Random)
        afferent_projection_dict["probability"].append(
            get_projection(projection)._connection_args[0]
        )
        ### size
        afferent_projection_dict["size"].append(len(get_projection(projection).pre))
        ### target type
        afferent_projection_dict["target"].append(get_projection(projection).target)

    return afferent_projection_dict


# def create_single_with_inputs_network(population_name, afferent_projection_dict):
#     ### create single neuron from population
#     single_neuron_with_inputs = Population(
#         1,
#         neuron=get_population(population_name).neuron_type,
#         name="single_neuron_with_inputs",
#     )
#     for attr_name, attr_val in get_population(population_name).init.items():
#         setattr(single_neuron_with_inputs, attr_name, attr_val)

#     ### create input populations
#     proj_spike_times_ampa_list = []
#     proj_spike_times_gaba_list = []
#     for idx_projection in range(len(afferent_projection_dict["projection_names"])):
#         spike_frequency = (
#             afferent_projection_dict["target firing rate"][idx_projection]
#             * afferent_projection_dict["probability"][idx_projection]
#             * afferent_projection_dict["size"][idx_projection]
#         )
#         print(
#             afferent_projection_dict["projection_names"][idx_projection],
#             spike_frequency,
#         )
#         spike_times = np.arange(0, 1000, 1 / spike_frequency)
#         if afferent_projection_dict["target"][idx_projection] == "ampa":

#             proj_spike_times_ampa_list.append(spike_times)
#         else:
#             proj_spike_times_gaba_list.append(spike_times)

#     ### create input populations, different neurons = different afferent projections
#     if len(proj_spike_times_ampa_list) > 0:
#         ampa_input_pop = SpikeSourceArray(
#             spike_times=proj_spike_times_ampa_list, name="single_neuron_ampa_inputs"
#         )
#     if len(proj_spike_times_gaba_list) > 0:
#         gaba_input_pop = SpikeSourceArray(
#             spike_times=proj_spike_times_gaba_list, name="single_neuron_gaba_inputs"
#         )

#     ### connect inputs to single neuron
#     if len(proj_spike_times_ampa_list) > 0:
#         ampa_input_proj = Projection(
#             pre=ampa_input_pop,
#             post=single_neuron_with_inputs,
#             target="ampa",
#             name="ampa_input_proj",
#         )
#         ampa_input_proj.connect_all_to_all(weights=0, force_multiple_weights=True)
#     if len(proj_spike_times_gaba_list) > 0:
#         gaba_input_proj = Projection(
#             pre=gaba_input_pop,
#             post=single_neuron_with_inputs,
#             target="gaba",
#             name="gaba_input_proj",
#         )
#         gaba_input_proj.connect_all_to_all(weights=0, force_multiple_weights=True)

#     ### create Monitor for single neuron
#     mon_single_with_inputs = Monitor(single_neuron_with_inputs, ["g_ampa", "g_gaba"])

#     ### create network with single neuron
#     net_single_with_inputs = Network()
#     net_single_with_inputs.add(
#         [
#             single_neuron_with_inputs,
#             mon_single_with_inputs,
#         ]
#     )
#     if len(proj_spike_times_ampa_list) > 0:
#         net_single_with_inputs.add(
#             [
#                 ampa_input_pop,
#                 ampa_input_proj,
#             ]
#         )
#     if len(proj_spike_times_gaba_list) > 0:
#         net_single_with_inputs.add(
#             [
#                 gaba_input_pop,
#                 gaba_input_proj,
#             ]
#         )
#     net_single_with_inputs.compile()

#     ### network dict
#     net_single_with_inputs_dict = {
#         "net": net_single_with_inputs,
#         "monitor": net_single_with_inputs.get(mon_single_with_inputs),
#         "population": net_single_with_inputs.get(single_neuron_with_inputs),
#     }
#     if len(proj_spike_times_ampa_list) > 0:
#         net_single_with_inputs_dict["ampa_input_proj"] = net_single_with_inputs.get(
#             ampa_input_proj
#         )
#     else:
#         net_single_with_inputs_dict["ampa_input_proj"] = None
#     if len(proj_spike_times_gaba_list) > 0:
#         net_single_with_inputs_dict["gaba_input_proj"] = net_single_with_inputs.get(
#             gaba_input_proj
#         )
#     else:
#         net_single_with_inputs_dict["gaba_input_proj"] = None
#     return net_single_with_inputs_dict


def get_mean_g_1000(spike_times_arr, spike_weights_arr, tau):
    ### for isi calculation append last spike at 1000ms
    isis_g_arr = np.diff(np.concatenate([spike_times_arr, np.array([1000])]))
    ### calc mean g
    mean_w = np.mean(spike_weights_arr)
    mean_isi = np.mean(isis_g_arr)
    mean_g = mean_w / ((1 / np.exp(-mean_isi / tau)) - 1)

    return mean_g


def get_g_values(afferent_projection_dict, tau_ampa, tau_gaba):
    ### get spike times over 1000 ms for ampa and gaba inputs based on afferent_projection_dict
    proj_spike_times_ampa_list = []
    proj_spike_weights_ampa_list = []
    proj_spike_times_gaba_list = []
    proj_spike_weights_gaba_list = []
    for idx_projection in range(len(afferent_projection_dict["projection_names"])):
        spike_frequency = (
            afferent_projection_dict["target firing rate"][idx_projection]
            * afferent_projection_dict["probability"][idx_projection]
            * afferent_projection_dict["size"][idx_projection]
        )
        spike_times = np.arange(0, 1, 1 / spike_frequency)
        spike_times = spike_times * 1000
        if afferent_projection_dict["target"][idx_projection] == "ampa":
            proj_spike_times_ampa_list.append(spike_times)
            proj_spike_weights_ampa_list.append(
                np.ones(len(spike_times))
                * afferent_projection_dict["weights"][idx_projection]
            )
        else:
            proj_spike_times_gaba_list.append(spike_times)
            proj_spike_weights_gaba_list.append(
                np.ones(len(spike_times))
                * afferent_projection_dict["weights"][idx_projection]
            )
    ### concatenate spike times of different projections
    ### ampa
    proj_spike_times_ampa_arr = np.concatenate(proj_spike_times_ampa_list)
    proj_spike_weights_ampa_arr = np.concatenate(proj_spike_weights_ampa_list)
    ### gaba
    proj_spike_times_gaba_arr = np.concatenate(proj_spike_times_gaba_list)
    proj_spike_weights_gaba_arr = np.concatenate(proj_spike_weights_gaba_list)
    ### sort the spike times and corresponding ids
    ### ampa
    sort_idx_ampa = np.argsort(proj_spike_times_ampa_arr)
    proj_spike_times_ampa_arr = proj_spike_times_ampa_arr[sort_idx_ampa]
    proj_spike_weights_ampa_arr = proj_spike_weights_ampa_arr[sort_idx_ampa]
    ### gaba
    sort_idx_gaba = np.argsort(proj_spike_times_gaba_arr)
    proj_spike_times_gaba_arr = proj_spike_times_gaba_arr[sort_idx_gaba]
    proj_spike_weights_gaba_arr = proj_spike_weights_gaba_arr[sort_idx_gaba]

    ### calculate mean g values
    ### ampa
    mean_ampa = get_mean_g_1000(
        spike_times_arr=proj_spike_times_ampa_arr,
        spike_weights_arr=proj_spike_weights_ampa_arr,
        tau=tau_ampa,
    )
    ### gaba
    mean_gaba = get_mean_g_1000(
        spike_times_arr=proj_spike_times_gaba_arr,
        spike_weights_arr=proj_spike_weights_gaba_arr,
        tau=tau_gaba,
    )

    return [mean_ampa, mean_gaba]


def get_g_of_single_proj(weight, proj_name, tau_ampa, tau_gaba):
    afferent_projection_dict["weights"] = []
    ### set only the given proj to 1
    for idx_proj in range(len(afferent_projection_dict["projection_names"])):
        if afferent_projection_dict["projection_names"][idx_proj] == proj_name:
            afferent_projection_dict["weights"].append(weight)
            proj_target_type = afferent_projection_dict["target"][idx_proj]
        else:
            afferent_projection_dict["weights"].append(0)
    g_ampa_val, g_gaba_val = get_g_values(afferent_projection_dict, tau_ampa, tau_gaba)
    if proj_target_type == "ampa":
        g_val = g_ampa_val
    else:
        g_val = g_gaba_val
    return g_val


def get_max_w_of_proj(proj_name, tau_ampa, tau_gaba, g_max):

    weight_list = [0]
    g_list = [0]
    init_weight_val = 1
    weight_val = init_weight_val
    g_val = 0
    alpha_tol = 0.001
    tolerance = g_max * alpha_tol
    n_it_max = 100
    n_it = 0
    ###
    while (
        not (0 <= (g_val - g_max) and (g_val - g_max) < tolerance) and n_it < n_it_max
    ):
        ### get f for I
        g_val = get_g_of_single_proj(
            weight=weight_val, proj_name=proj_name, tau_ampa=tau_ampa, tau_gaba=tau_gaba
        )
        ### append I_app and f_rec to f_list/I_list
        g_list.append(g_val)
        weight_list.append(weight_val)
        # print(g_list)
        # print(weight_list)
        ### predict new I_app for f_max
        weight_val = predict_1d(X=g_list, y=weight_list, X_pred=g_max)
        ### increase iterator
        n_it += 1
    print(np.array([weight_list, g_list]).T)

    max_w = weight_list[-1]
    return max_w


def get_w_max(afferent_projection_dict, population_name, g_ampa_max, g_gaba_max):

    tau_ampa = get_population(population_name).tau_ampa
    tau_gaba = get_population(population_name).tau_gaba

    ### loop over afferent projections
    print(f"g_ampa_max: {g_ampa_max}, g_gaba_max: {g_gaba_max}")
    afferent_projection_dict["max_weight"] = []
    for idx_proj in range(len(afferent_projection_dict["projection_names"])):
        print(afferent_projection_dict["projection_names"][idx_proj])
        ### find max weight for projection
        proj_target_type = afferent_projection_dict["target"][idx_proj]
        if proj_target_type == "ampa":
            g_max = g_ampa_max
        else:
            g_max = g_gaba_max
        afferent_projection_dict["max_weight"].append(
            get_max_w_of_proj(
                proj_name=afferent_projection_dict["projection_names"][idx_proj],
                tau_ampa=tau_ampa,
                tau_gaba=tau_gaba,
                g_max=g_max,
            )
        )
    ### remove weight key from afferent_projection_dict which was added during the process
    afferent_projection_dict.pop("weights")


def get_g_values_from_weight_dict(
    weight_dict, population_name, afferent_projection_dict
):

    weight_dict_of_pop = weight_dict[population_name]
    afferent_projection_dict["weights"] = [0] * len(
        afferent_projection_dict["projection_names"]
    )
    for proj_name, weight_val in weight_dict_of_pop.items():
        proj_idx = afferent_projection_dict["projection_names"].index(proj_name)
        afferent_projection_dict["weights"][proj_idx] = weight_val

    tau_ampa = get_population(population_name).tau_ampa
    tau_gaba = get_population(population_name).tau_gaba

    g_ampa, g_gaba = get_g_values(afferent_projection_dict, tau_ampa, tau_gaba)

    return [g_ampa, g_gaba]


def check_and_rescale_weights_of_pop(
    weight_dict, population_name, afferent_projection_dict
):
    g_ampa, g_gaba = get_g_values_from_weight_dict(
        weight_dict=weight_dict,
        population_name=population_name,
        afferent_projection_dict=afferent_projection_dict,
    )

    ### check if they exceed g_max, if yes --> rescale
    g_ampa_factor = g_ampa / g_ampa_max
    g_gaba_factor = g_gaba / g_gaba_max
    if g_ampa_factor > 1:
        ### rescale all ampa weights
        for proj_idx in range(len(afferent_projection_dict["projection_names"])):
            if afferent_projection_dict["target"][proj_idx] == "ampa":
                proj_name = afferent_projection_dict["projection_names"][proj_idx]
                weight_dict[population_name][proj_name] *= 1 / g_ampa_factor
    if g_gaba_factor > 1:
        ### rescale all ampa weights
        for proj_idx in range(len(afferent_projection_dict["projection_names"])):
            if afferent_projection_dict["target"][proj_idx] == "gaba":
                proj_name = afferent_projection_dict["projection_names"][proj_idx]
                weight_dict[population_name][proj_name] *= 1 / g_gaba_factor

    if g_ampa_factor > 1 or g_gaba_factor > 1:
        g_ampa, g_gaba = get_g_values_from_weight_dict(
            weight_dict=weight_dict,
            population_name=population_name,
            afferent_projection_dict=afferent_projection_dict,
        )

    return [weight_dict, g_ampa, g_gaba]


if __name__ == "__main__":
    #######   PARAMETERS   ######
    params = {}
    #######   POPULATIONS PARAMETERS   ######
    ### cortex / input populations
    params["cor.size"] = 40
    params["cor.tau_up"] = 10
    params["cor.tau_down"] = 30
    ### BG Populations
    params["stn.size"] = 10
    params["stn.a"] = 0.005
    params["stn.b"] = 0.265
    params["stn.c"] = -65
    params["stn.d"] = 2
    params["stn.n2"] = 0.04
    params["stn.n1"] = 5
    params["stn.n0"] = 140
    params["stn.tau_ampa"] = 10
    params["stn.tau_gaba"] = 10
    params["stn.E_ampa"] = 0
    params["stn.E_gaba"] = -90
    params["snr.size"] = 20
    params["snr.a"] = 0.005
    params["snr.b"] = 0.585
    params["snr.c"] = -65
    params["snr.d"] = 4
    params["snr.n2"] = 0.04
    params["snr.n1"] = 5
    params["snr.n0"] = 140
    params["snr.tau_ampa"] = 10
    params["snr.tau_gaba"] = 10
    params["snr.E_ampa"] = 0
    params["snr.E_gaba"] = -90
    params["gpe.size"] = 25
    params["gpe.a"] = 0.039191890241715294
    params["gpe.b"] = 0.000548238111291427
    params["gpe.c"] = -49.88014418530518
    params["gpe.d"] = 108.0208225074675
    params["gpe.n2"] = 0.08899515481507077
    params["gpe.n1"] = 1.1929776239208976
    params["gpe.n0"] = 24.2219699019072
    params["gpe.tau_ampa"] = 10
    params["gpe.tau_gaba"] = 10
    params["gpe.E_ampa"] = 0
    params["gpe.E_gaba"] = -90
    params["thal.size"] = 10
    params["thal.a"] = 0.02
    params["thal.b"] = 0.2
    params["thal.c"] = -65
    params["thal.d"] = 6
    params["thal.n2"] = 0.04
    params["thal.n1"] = 5
    params["thal.n0"] = 140
    params["thal.tau_ampa"] = 10
    params["thal.tau_gaba"] = 10
    params["thal.E_ampa"] = 0
    params["thal.E_gaba"] = -90
    #######   PROJECTIONS PARAMETERS   ######
    params["cor__stn.probability"] = 0.2
    params["stn__snr.probability"] = 0.3
    params["stn__gpe.probability"] = 0.35
    params["gpe__stn.probability"] = 0.35
    params["gpe__snr.probability"] = 0.4
    params["snr__thal.probability"] = 0.6
    params["snr__snr.probability"] = 0.6

    ### create model which should be configurated
    setup(dt=0.1)
    model = generate_model(
        model_creation_function=BGM_part_function,
        model_kwargs={"params": params},
        name="BGM_part_model",
        description="Part of a BGM circuit",
        do_create=False,
    )

    ### model configurator should get target firing rates as input
    target_firing_rate_dict = {
        "cor": 15,
        "stn": 30,
        "gpe": 50,
        "snr": 60,
        "thal": 5,
    }

    ### the model configurator also needs the model CompNeuroPy object
    ### clear ANNarchy and create model
    cnp_clear()
    model.create(do_compile=False)

    ### create node for one population e.g. SNr
    ### get all afferent populations/projections
    population_name = "snr"

    ### get afferent projection dict
    afferent_projection_dict = get_afferent_projection_dict(
        model=model, population_name=population_name
    )

    ### get max values for I_app, g_ampa and g_gaba
    ### first create single neuron network
    ### before single_neuron also create many_neuron for later
    net_many_dict = create_many_network(model=model, population_name=population_name)
    net_single_dict = create_single_network(population_name=population_name)

    ### get max values for I_app, g_ampa and g_gaba, and initial variable values
    prepare_list = prepare_f_I_g_curve(
        net_single_dict=net_single_dict,
        f_t=target_firing_rate_dict[population_name],
        population_name=population_name,
    )
    I_max, g_ampa_max, g_gaba_max, variable_init_dict = prepare_list

    ### with g_ampa_max and g_gaba_max get the max_w values for the afferent projections
    get_w_max(
        afferent_projection_dict=afferent_projection_dict,
        population_name=population_name,
        g_ampa_max=g_ampa_max,
        g_gaba_max=g_gaba_max,
    )
    print(afferent_projection_dict)
    return_dict_for_pop = {
        proj_name: afferent_projection_dict["max_weight"][proj_idx]
        for proj_idx, proj_name in enumerate(
            afferent_projection_dict["projection_names"]
        )
    }

    max_w_return_dict = {"snr": return_dict_for_pop}
    print("max_w_return_dict")
    print(max_w_return_dict)

    ### now the user has max w (so far without projection conditions)
    ### return the user a dict, keys= population name, values = dicts with keys = afferent projections, vals = max_w
    ### now the user selects w and the mean_base should beautomatically found
    ### the user should give the weights in the exact form as the max weights were returne (dict of dicts)
    weight_dict = {
        "pop_1": {"aff_proj_11": 10, "aff_proj_12": 5},
        "pop_2": {"aff_proj_21": 1, "aff_proj_22": 0.5},
        "snr": {
            "stn__snr": 6.254214832726668,
            "gpe__snr": 0.5124776746963882,
            "snr__snr": 0.355887274094714,
        },
    }

    weight_dict, g_ampa, g_gaba = check_and_rescale_weights_of_pop(
        weight_dict=weight_dict,
        population_name=population_name,
        afferent_projection_dict=afferent_projection_dict,
    )

    ### get f_I_g_curve and I_f_g_curve with many_neuron network
    f_I_g_curve, I_f_g_curve, I_max, g_ampa_max = get_f_I_g_curve(
        net_many_dict=net_many_dict,
        prepare_list=prepare_list,
    )

    ### predict I from g_ampa, g_gaba and target firing rate using I_f_g_curve
    I_pred = I_f_g_curve(
        p1=g_ampa, p2=g_gaba, p3=target_firing_rate_dict[population_name]
    )[0]
    print(f_I_g_curve.X_name_dict)
    print(I_f_g_curve.X_name_dict)
    print(f"I(f=[100, 50, 25])={I_f_g_curve(p3=[100,50,25])}")
    print(
        f"I_pred(f=f_t={target_firing_rate_dict[population_name]}, g_ampa={g_ampa}, g_gaba={g_gaba})={I_pred}"
    )

    f_rec = get_rate_1000(
        net=net_single_dict["net"],
        population=net_single_dict["population"],
        variable_init_dict=variable_init_dict,
        monitor=net_single_dict["monitor"],
        I_app=I_pred,
        g_ampa=g_ampa,
        g_gaba=g_gaba,
    )
    print(f"f_rec(I=f_pred={I_pred}, g_ampa={g_ampa}, g_gaba={g_gaba})={f_rec}")
    print(f"== f_t ({target_firing_rate_dict[population_name]})?")

    ### test more predicted Is
    I_pred = I_f_g_curve(
        p1=[g_ampa * fac for fac in np.linspace(1, 0.1, 10)],
        p2=[g_gaba * fac for fac in np.linspace(1, 0.1, 10)],
        p3=[target_firing_rate_dict[population_name]] * 10,
    )

    f_rec_list = []
    for idx in range(10):
        f_rec = get_rate_1000(
            net=net_single_dict["net"],
            population=net_single_dict["population"],
            variable_init_dict=variable_init_dict,
            monitor=net_single_dict["monitor"],
            I_app=I_pred[idx],
            g_ampa=[g_ampa * fac for fac in np.linspace(1, 0.1, 10)][idx],
            g_gaba=[g_gaba * fac for fac in np.linspace(1, 0.1, 10)][idx],
        )
        f_rec_list.append(f_rec)
    print(f"f_rec == {target_firing_rate_dict[population_name]}?: {f_rec_list}")

    ### get f(I)
    I_arr = np.linspace(-I_max, I_max, 100)
    f_arr = np.clip(f_I_g_curve(p1=I_arr), 0, None)
    plt.figure()
    plt.plot(I_arr, f_arr)
    plt.xlabel("I")
    plt.ylabel("f [Hz]")
    plt.savefig("tmp_f_I.png")
    plt.close("all")

    ### get f(g_ampa)
    g_ampa_arr = np.linspace(0, g_ampa_max, 100)
    f_arr = np.clip(f_I_g_curve(p2=g_ampa_arr), 0, None)
    plt.figure()
    plt.plot(g_ampa_arr, f_arr)
    plt.xlabel("g_ampa")
    plt.ylabel("f [Hz]")
    plt.savefig("tmp_f_g_ampa.png")
    plt.close("all")

    ### after all clear ANNarchy and create model
    cnp_clear()
    model.create(do_compile=False)
