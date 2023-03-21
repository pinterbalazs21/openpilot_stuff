import numpy as np
import onnx
import onnxruntime as rt

class InputWrapper:
    def __init__(self, session, input_imgs, wide_input_imgs, desire=None, traffic_convention=None, initial_state=None):
        self.session = session
        self.input_imgs = input_imgs
        self.wide_input_imgs = wide_input_imgs
        
        if desire is None:
            desire_data = np.array([0]).astype('float32')
            desire_data.resize((1,8))
            self.desire = desire_data
        else:
            self.desire = desire

        if traffic_convention is None:
            traffic_convention_data = np.array([0]).astype('float32')
            traffic_convention_data.resize((1,2))
            self.traffic_convention = traffic_convention_data
        else:
            self.traffic_convention = traffic_convention

        if initial_state is None:
            initial_state_data = np.array([0]).astype('float32')
            initial_state_data.resize((1,512))
            self.initial_state = initial_state_data
        else:
            self.initial_state = initial_state

    def get_model_input(self):
        input_names = [input.name for input in self.session.get_inputs()]
        input_data = {
            input_names[0]: self.input_imgs,
            input_names[1]: self.wide_input_imgs,
            input_names[2]: self.desire,
            input_names[3]: self.traffic_convention,
            input_names[4]: self.initial_state,
        }
    
        # Print the shapes of the inputs
        #for name, data in input_data.items():
        #    print(f"{name}: {data.shape}")
    
        return input_data
