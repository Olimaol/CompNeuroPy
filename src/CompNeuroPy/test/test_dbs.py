from multiprocessing import Process, Manager


def import_dbs_example(_, return_dict):
    from CompNeuroPy.examples.dbs_stimulator import main

    return_dict[0] = main()


def test_dbs():
    return_dict = Manager().dict()
    proc = Process(target=import_dbs_example, args=(0, return_dict))
    proc.start()
    proc.join()

    assert 1 == return_dict[0]


if __name__ == "__main__":
    test_dbs()
