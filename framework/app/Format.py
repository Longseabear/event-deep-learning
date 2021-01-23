from framework.app.app import App, SingletoneInstance
from pyparsing import nestedExpr
import re

class Formatter():
    def __init__(self, controller, contents={}):
        self.app = App.instance()
        self.model_controller = controller
        self.contents = contents
        pass

    def StateFormatting(self, variable):
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

    def MainStateBasedFormatting(self, format='[$main:step:05]e_[$main:step:05]s_[$content]_[$now:step:05].[$format]'):
        try:
            out = ''
            formats = [self.StateFormatting(i) for i in re.split(r"\[(.*?)\]", format) if len(i)>0]
            for i in formats:
                out += i
            return out
        except Exception as e:
            raise e