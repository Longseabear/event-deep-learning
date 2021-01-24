from framework.app.app import App, SingletoneInstance
from pyparsing import nestedExpr
import re

class Formatter():
    def __init__(self, controller, contents={}, format=''):
        """
        Manages various formats.
        This class is dependent on the current Controller and configuration safe.

        :param controller: Main Controller
        :param contents: Variables dictionary
        :param format: The format variable must be specified in [$name] format.
        example:
            f = Formatter(Controller, contents={'content':'model', 'format':'exe'}, format='[$content]_checkpoint.[$format]')
            print(f.Formatting())
            -> model_checkpoint.exe
        """
        self.model_controller = controller
        self.contents = contents
        self.format = format

    def StateFormatting(self, variable):
        """
        Create a string according to the given format. Variables are separated by $.
        If the variable contains :, it means that the variable of state is printed.
        it consist of [$STATE_NAME:VARIABLE_NAME:ZFILL]
        If it is $main:step:05, the step variable of the main state is output as zfill(5).

        STATE_NAME: main or now
        VARIABLE_NAME: state variable name
        ZFILL : How many zeros to fill

        :param variable:
        :return:
        """
        try:
            format = variable.split(':')
            if len(format)==1:
                return self.contents[variable[1:]] if variable[0]=='$' else variable
            state = format[0][1:]
            var = format[1]
            zfill = 1
            if len(format)>2:
                zfill = int(format[2])
            if state=='main':
                model_state = self.model_controller.get_main_state()
            elif state=='now':
                model_state = self.model_controller.get_state()
            else:
                raise NotImplementedError
            return str(model_state.__getattribute__(var)).zfill(zfill)
        except Exception as e:
            raise e

    def Formatting(self) -> str:
        """
        Assign variables according to format. return string
        :return: formatted str
        """
        try:
            out = ''
            formats = [self.StateFormatting(i) for i in re.split(r"\[(.*?)\]", self.format) if len(i)>0]
            for i in formats:
                out += str(i)
            return out
        except Exception as e:
            raise e

    def MakeContentsFromString(self, string: str) -> None:
        """
        Extracts a variable according to the format from the given string.
        The outside of the format must be independent.
        :param string:
        :return:
        """
        try:
            formats = [i for i in re.split(r"\[(.*?)\]", self.format) if len(i)>0]
            comparator = [f for f in formats if f[0] != '$']
            formatter = [f for f in formats if f[0] == '$']
            count = 0
            for comp in comparator:
                self.contents[formatter[count][1:]] = string[:string.find(comp)]
                string = string[string.find(comp)+len(comp):]
                count += 1
            self.contents[formatter[count][1:]] = string

        except Exception as e:
            raise e

    def CheckSameBaseName(self, f1, f2) -> bool:
        """
        Check if two elements are the same except for format
        :param f1: string 1
        :param f2: string 2
        :return: true if all elements except format are the same

        """
        try:
            formats = [i for i in re.split(r"\[(.*?)\]", self.format) if len(i)>0]
            comparator = [f for f in formats if f[0] != '$']
            return all([i in f1 and i in f2 for i in comparator])
        except Exception as e:
            raise e

class MainStateBasedFormatter(Formatter):
    def __init__(self, controller, contents={}, format=''):
        """
        Formatter's wrapper function. if main state == current state, all current state name replace to main.
        """
        if controller.get_main_state() == controller.get_state():
            format.replace('now:', 'main:')
        super().__init__(controller, contents, format)

if __name__ == '__main__':
    f = Formatter(None, contents={'content':'cdef','batch':523,'format':'exe'},format='[$content]_[$batch]_checkpoint.[$format]')
    a = f.Formatting()
    f = Formatter(None, contents={'content':'asdasd','batch':523,'format':'exe'},format='[$content]_[$batch]_checkpoint.[$format]')
    b = f.Formatting()
    print(f.CheckSameBaseName(a,b))
