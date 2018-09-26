from input_run_handler import *


class Ann():
    # Class for keeping track of the case manager and the models.
    def set_cman(self, cman):
        self.cman = cman

    def get_cman(self):
        return self.cman

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model


def main():
    ann = Ann()
    irh = InputRunHandler(ann)

    while True:
        u_input = input("Enter command: ")

        if u_input == "q":
            break
        elif u_input == "h":
            print("We support the following commands: ")
            print("Nothing at the moment")
        else:
            irh.evaluate_input(u_input)


if __name__ == "__main__":
    main()
