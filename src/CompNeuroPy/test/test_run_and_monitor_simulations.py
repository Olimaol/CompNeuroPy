from multiprocessing import Process, Manager


def import_run_and_monitor_simulations(_, return_dict):
    from CompNeuroPy.examples.run_and_monitor_simulations import (
        main as run_and_monitor_simulations,
    )

    return_dict[0] = run_and_monitor_simulations()


def test_run_and_monitor_simulations():
    return_dict = Manager().dict()
    proc = Process(target=import_run_and_monitor_simulations, args=(0, return_dict))
    proc.start()
    proc.join()

    assert 1 == return_dict[0]


if __name__ == "__main__":
    test_run_and_monitor_simulations()
