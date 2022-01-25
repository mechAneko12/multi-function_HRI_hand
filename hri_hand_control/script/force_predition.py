import time
import pickle 
from keras.models import load_model
import numpy as np
from pathlib import Path

class force_predictor:
    def __init__(self, dataset_name, MIN=0, MAX=0.012, velocity=0.001):
        self.fingers_state = {'index_control' : None,
                             'middle_control' : None,
                             'ring_control' : None,
                             'little_control' : None,
                             'thumb_joint_control' : None,
                             'thumb_control' : None
                            }
        self.reset()
                
        self.MIN = MIN
        self.MAX = MAX
        self.velocity = velocity
        
        path = Path(__file__).parent / dataset_name + '/trained_model/'
        
        with open(path / 'scaler_x_middle.pickle', mode='rb') as fp:
            self.scaler_x_middle = pickle.load(fp)
        with open(path / 'scaler_y_middle.pickle', mode='rb') as fp:
            self.scaler_y_middle = pickle.load(fp)

        self.reg_middle = load_model(path / 'reg_middle.h5')

        self.feature_tmp = None

    def __call__(self, hand, feature_vector):
        self.feature_tmp = feature_vector
        self._switch(hand)
        return self.fingers_state

    def _switch(self, hand):
        if hand == 1:
            pass
            #self._flex()
        elif hand == 2:
            self._flex(['middle_control'], self.scaler_x_middle, self.scaler_y_middle, self.reg_middle)
        elif hand == 3:
            pass
            #self._flex()
        elif hand == 4:
            self.all_ext()
        else:
            pass

    def reset(self):
        for i, m in self.fingers_state.items():
            self.fingers_state[i] = 0
    
    def all_ext(self):
        for i, m in self.fingers_state.items():
            if self.fingers_state[i] > self.MIN:
                self.fingers_state[i] -= self.velocity
    
    def _flex(self, flexion_finger_list, scaler_x, scaler_y, reg_model):
        for i, m in self.fingers_state.items():
            if i in flexion_finger_list:
                if self.fingers_state[i] < self.MAX:
                    
                    feature_tmp = scaler_x(self.feature_tmp)
                    pred_f = reg_model(feature_tmp)
                    pred_f_tmp = np.mean(scaler_y.inverse_transform(pred_f) * 0.012 / scaler_y.data_max_)
                    
                    print(pred_f_tmp)
                    if pred_f > self.MAX:
                        pred_f_tmp = self.MAX
                    self.fingers_state[i] = pred_f_tmp
            else:
                if self.fingers_state[i] > self.MIN:
                    self.fingers_state[i] -= self.velocity


class control:
    def __init__(self, hj_tf, sleep_time = 0.04):
        self.hj_tf = hj_tf
        self.stm_index_id = 0x01
        self.stm_middle_id = 0x02
        self.stm_ring_id = 0x03
        self.stm_little_id = 0x04
        self.stm_thumb_id = 0x05
        self.stm_thumb_joint_id = 0x06
        self.sleep_time = sleep_time

    def move(self, fingers_state):
        index_pris_val = fingers_state['index_control']
        middle_pris_val = fingers_state['middle_control']
        ring_pris_val = fingers_state['ring_control']
        little_pris_val = fingers_state['little_control']
        thumb_pris_val = fingers_state['thumb_joint_control']
        thumb_joint_pris_val = fingers_state['thumb_control']

        # index
        self.hj_tf.index_control(index_pris_val)
        self.hj_tf.hj_finger_control(self.stm_index_id, index_pris_val)

        # middle
        self.hj_tf.middle_control(middle_pris_val)
        self.hj_tf.hj_finger_control(self.stm_middle_id, middle_pris_val)

        # ring
        self.hj_tf.ring_control(ring_pris_val)
        self.hj_tf.hj_finger_control(self.stm_ring_id, ring_pris_val)

        # little
        self.hj_tf.little_control(little_pris_val)
        self.hj_tf.hj_finger_control(self.stm_little_id, little_pris_val)

        # thumb_joint
        self.hj_tf.thumb_joint_control(thumb_joint_pris_val)
        self.hj_tf.hj_finger_control(self.stm_thumb_joint_id, thumb_joint_pris_val)

        # thumb
        self.hj_tf.thumb_control(thumb_pris_val)
        self.hj_tf.hj_finger_control(self.stm_thumb_id, thumb_pris_val)

        #time.sleep(self.sleep_time)



