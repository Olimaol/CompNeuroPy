from multiprocessing import Process, Manager


def import_create_model(_, return_dict):
    from CompNeuroPy.examples.create_model import main as create_model

    return_dict[0] = create_model()


def test_create_model():
    return_dict = Manager().dict()
    proc = Process(target=import_create_model, args=(0, return_dict))
    proc.start()
    proc.join()

    assert 1 == return_dict[0]


if __name__ == "__main__":
    test_create_model()
