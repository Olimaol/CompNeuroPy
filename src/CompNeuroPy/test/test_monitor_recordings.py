from multiprocessing import Process, Manager


def import_monitor_recordings(_, return_dict):
    from CompNeuroPy.examples.monitor_recordings import main as monitor_recordings

    return_dict[0] = monitor_recordings()


def test_monitor_recordings():
    return_dict = Manager().dict()
    proc = Process(target=import_monitor_recordings, args=(0, return_dict))
    proc.start()
    proc.join()

    assert 1 == return_dict[0]


if __name__ == "__main__":
    test_monitor_recordings()
