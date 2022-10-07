def test_simulate_monitor_plot():
    from CompNeuroPy.examples.run_and_monitor_simulations import (
        main as simulate_monitor,
    )
    from CompNeuroPy.examples.plot_recordings import main as plot

    assert 1 == simulate_monitor()
    assert 1 == plot()
