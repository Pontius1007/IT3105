from input_run_handler import *
import sys

from colorama import init
init(strip=not sys.stdout.isatty()) # strip colors if stdout is redirected
from termcolor import cprint
from pyfiglet import figlet_format


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
    cprint(figlet_format('AI is the future', font='doom'),
           'yellow', attrs=['bold'])

    while True:
        print("\n Welcome to our project. Press 'h' for help. \n")
        u_input = input("Enter command: ")

        if u_input == "q":
            break
        elif u_input == "h":
            print("\n We support the following commands: ")
            print("1. 'r' or 'run'. This will start running/training. ")
            print("2. 'q' will quit the application or the input question. ")
            print("3. 'load json' or 'lj'. This will load new parameters from a JSON file. ")
            print("4. 'predict' or 'p'. This will execute the predict function.")
            print("5. 'show' or 'plt'. This will take over your cmd and put the graphs in interactive mode")
            print("6. The list of datasets we support:\n "
                  "bitcounter \n"
                  "autoencoder \n"
                  "parity \n"
                  "symmetry \n"
                  "segmentcounter \n"
                  "yeast \n"
                  "glass \n"
                  "wine \n"
                  "iris \n"
                  "mnist \n")
        else:
            irh.evaluate_input(u_input)


if __name__ == "__main__":
    main()
