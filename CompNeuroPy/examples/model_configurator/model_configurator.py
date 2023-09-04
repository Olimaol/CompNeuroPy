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
def train_regr(X_raw, y_raw):
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

    return f_I_g(regr_model, scaler_X, scaler_y)


class f_I_g:
    def __init__(self, regr_model, scaler_X, scaler_y) -> None:
        self.regr_model = regr_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def __call__(self, I_app=None, g_ampa=None, g_gaba=None):
        """
        params:
            I_app, number or array, default 0
            g_ampa, number or array, default 0
            g_gaba, number or array, default 0

            if I_app, g_ampa, g_gaba == array --> same size!

        return:
            f_arr
        """
        ### check which values are given
        I_app_given = not (isinstance(I_app, type(None)))
        g_ampa_given = not (isinstance(g_ampa, type(None)))
        g_gaba_given = not (isinstance(g_gaba, type(None)))

        ### reshape all to target shape
        I_app = np.array(I_app).reshape((-1, 1)).astype(float)
        g_ampa = np.array(g_ampa).reshape((-1, 1)).astype(float)
        g_gaba = np.array(g_gaba).reshape((-1, 1)).astype(float)

        ### check if arrays of given values have same size
        size_arr = np.array([I_app.shape[0], g_ampa.shape[0], g_gaba.shape[0]])
        given_arr = np.array([I_app_given, g_ampa_given, g_gaba_given]).astype(bool)
        if len(size_arr[given_arr]) > 0:
            all_size = size_arr[given_arr][0]
            all_same_size = np.all(size_arr[given_arr] == all_size)
            if not all_same_size:
                raise ValueError(
                    "f_I_g call: given I_app, g_ampa and g_gaba have to have same size!"
                )

        ### set the correctly sized arrays for the not given values
        if not I_app_given:
            I_app = np.zeros(all_size).reshape((-1, 1))
        if not g_ampa_given:
            g_ampa = np.zeros(all_size).reshape((-1, 1))
        if not g_gaba_given:
            g_gaba = np.zeros(all_size).reshape((-1, 1))

        ### predict f_arr
        X_raw = np.concatenate([I_app, g_ampa, g_gaba], axis=1)
        f_arr = pred_regr(X_raw, self.regr_model, self.scaler_X, self.scaler_y)
        f_arr = np.clip(f_arr, 0, None)

        return f_arr


def pred_regr(X_raw, regr_model, scaler_X, scaler_y):
    """
    X shape = (n_samples, n_features)
    """

    X_scaled = scaler_X.transform(X_raw)
    pred_scaled = regr_model.predict(X_scaled)
    pred_raw = scaler_y.inverse_transform(pred_scaled[:, None])

    return pred_raw


def get_rate_1000(
    net, population, variable_init_dict, monitor, I_app=0, g_ampa=0, g_gaba=0
):
    """
    simulates 1000 ms a population consisting of a single neuron and returns the firing rate
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


def predict_1d(X, y, X_pred):
    I_f = interp1d(x=X, y=y, fill_value="extrapolate")
    y_pred = I_f(X_pred)
    return y_pred


def get_init_neuron_variables(net, population):
    net.reset()
    population.I_app = 0
    net.simulate(2000)
    variable_init_dict = {
        var_name: getattr(population, var_name) for var_name in population.variables
    }
    return variable_init_dict


def prepare_f_I_g_curve(net, population, monitor, f_t, population_name):

    ### get initialization of neuron
    variable_init_dict = get_init_neuron_variables(net, population)

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
    alpha_tol = 0.02
    tolerance = (f_max - f_0) * alpha_tol
    n_it_max = 100
    n_it = 0
    I_app = init_I_app
    I_list = [0]
    f_list = [f_0]
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
    alpha_tol = 0.02
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
    alpha_tol = 0.02
    tolerance = (f_0 - f_max) * alpha_tol
    n_it_max = 100
    n_it = 0
    g_gaba = init_g_gaba
    g_gaba_list = [0]
    f_list = [f_max]
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


def get_f_I_g_curve(net_single_dict, net_many_dict, f_t, population_name):
    ### get max values for I_app, g_ampa and g_gaba, and initial variable values
    I_max, g_ampa_max, g_gaba_max, variable_init_dict = prepare_f_I_g_curve(
        net_single_dict["net"],
        net_single_dict["population"],
        net_single_dict["monitor"],
        f_t,
        population_name,
    )

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

    ### get f for I
    f_rec_arr = get_rate_1000(
        net_many_dict["net"],
        net_many_dict["population"],
        variable_init_dict,
        net_many_dict["monitor"],
        g_ampa=g_ampa_arr,
        g_gaba=g_gaba_arr,
        I_app=I_app_arr,
    )

    f_I_g_data_arr = np.concatenate([I_g_arr, f_rec_arr[:, None]], axis=1)

    print("train regr...")
    f_I_g_curve = train_regr(
        X_raw=f_I_g_data_arr[:, :3], y_raw=f_I_g_data_arr[:, 3][:, None]
    )

    return [f_I_g_curve, I_max, g_ampa_max]


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
    if not (model.created):
        model.create(do_compile=False)

    ### create node for one population e.g. SNr
    ### get all afferent populations/projections
    population_name = "snr"

    afferent_projection_dict = {}
    afferent_projection_dict["projection names"] = []
    for projection in model.projections:
        if get_projection(projection).post.name == population_name:
            afferent_projection_dict["projection names"].append(projection)

    ### get target firing rates resting-state for afferent projections
    afferent_projection_dict["target firing rate"] = []
    afferent_projection_dict["probability"] = []
    afferent_projection_dict["size"] = []
    for projection in afferent_projection_dict["projection names"]:
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

    ### get f(I) and I(g_ampa,g_gaba,base_mean)

    ### create many neuron population whose simulation of 1000ms takes about 10sec
    nr_many_neurons = get_nr_many_neurons(model, population_name)
    many_neuron = Population(
        nr_many_neurons,
        neuron=get_population(population_name).neuron_type,
        name="many_neuron",
    )
    for attr_name, attr_val in get_population(population_name).init.items():
        setattr(many_neuron, attr_name, attr_val)

    ### create Monitor for many neuron
    mon_many = Monitor(many_neuron, ["spike", "g_ampa", "g_gaba"])

    ### create network with many neuron
    net_many = Network()
    net_many.add([many_neuron, mon_many])
    net_many.compile()

    ### create single neuron from population
    single_neuron = Population(
        1, neuron=get_population(population_name).neuron_type, name="single_neuron"
    )
    for attr_name, attr_val in get_population(population_name).init.items():
        setattr(single_neuron, attr_name, attr_val)

    ### create Monitor for single neuron
    mon_single = Monitor(single_neuron, ["spike", "g_ampa", "g_gaba", "I"])

    ### create network with single neuron
    net_single = Network()
    net_single.add([single_neuron, mon_single])
    net_single.compile()

    f_I_g_curve, I_max, g_ampa_max = get_f_I_g_curve(
        net_single_dict={
            "net": net_single,
            "population": net_single.get(single_neuron),
            "monitor": net_single.get(mon_single),
        },
        net_many_dict={
            "net": net_many,
            "population": net_many.get(many_neuron),
            "monitor": net_many.get(mon_many),
        },
        f_t=target_firing_rate_dict[population_name],
        population_name=population_name,
    )

    ### get f(I)
    I_arr = np.linspace(-I_max, I_max, 100)
    f_arr = f_I_g_curve(I_app=I_arr)
    plt.figure()
    plt.plot(I_arr, f_arr)
    plt.xlabel("I")
    plt.ylabel("f [Hz]")
    plt.savefig("tmp_f_I.png")
    plt.close("all")

    ### get f(g_ampa)
    g_ampa_arr = np.linspace(0, g_ampa_max, 100)
    f_arr = f_I_g_curve(g_ampa=g_ampa_arr)
    plt.figure()
    plt.plot(g_ampa_arr, f_arr)
    plt.xlabel("g_ampa")
    plt.ylabel("f [Hz]")
    plt.savefig("tmp_f_g_ampa.png")
    plt.close("all")

    ### remove all networks etc and compile the model
    cnp_clear()
    model.create()
