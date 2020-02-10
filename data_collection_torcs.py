import pickle
import keyboard
import torch
from .torcs_supervised_learning import Supervised_Learning_Agent
from .helpers import count_down, give_a_frame

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    Tensor = torch.Tensor
    LongTensor = torch.LongTensor

    agent = Supervised_Learning_Agent(state_size=(70, 70), action_size=5, test_mode=False)
    # agent.load("Car_Racing_weights1.pth")

    action_size = 5
    batch_size = 32
    n_episodes = 300
    episode_length = 5000
    update_frequency_target = 50
    update_frequency_main = 1
    hidden_layer = 512
    save_frequency = 1000

    total_time = 0

    count_down()
    with open("agents_experiences_data.pickle", "wb") as pickle_out:

        for episode in range(n_episodes):

            for time in range(episode_length):  # max time, increase this number later

                state = give_a_frame(normalize=True)
                # ------------------------RENDERING-------------------------
                # render(frame=state)
                # -----------------------------------------------------------

                if keyboard.is_pressed('w'):  # if key 'space' is pressed
                    action_user_index = 0
                    print("W key has been pressed")
                elif keyboard.is_pressed('a'):
                    action_user_index = 1
                    print("A key has been pressed")
                elif keyboard.is_pressed('d'):
                    action_user_index = 2
                    print("D key has been pressed")
                elif keyboard.is_pressed('s'):
                    action_user_index = 3
                    print("S key has been pressed")
                else:
                    action_user_index = 4
                    print("Nothing has been pressed")

                agent.remember(state, action_user_index, total_time)
                if total_time == 1000:
                    pickle.dump(agent.memory, pickle_out)

                print("Total Time: ", total_time)

                total_time += 1
