import torch
from .directkeys import PressKey, ReleaseKey, W, A, S, D
from .helpers import give_a_frame, count_down
from .torcs_supervised_learning import Supervised_Learning_Agent

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    Tensor = torch.Tensor
    LongTensor = torch.LongTensor

    agent = Supervised_Learning_Agent(state_size=(70, 70, 1), action_size=5, deterministic_action=False)

    agent.load("Torch_weights19.pth")

    total_time = 0
    counter = 0
    count_down()
    while True:

        frame = give_a_frame(normalize=True)
        # ------------------------RENDERING-------------------------
        # render(frame=frame)
        # -----------------------------------------------------------
        if total_time % 2 == 0:
            # state = torch.from_numpy(np.moveaxis(frame,-1,0))
            state = torch.from_numpy(frame)
            state = state.to(device, dtype=torch.float32)
            state = state.unsqueeze(0).unsqueeze(0)
            action, action_index, q_values = agent.act(state)

            if action_index is 0:
                ReleaseKey(A)
                ReleaseKey(D)
                ReleaseKey(S)

                PressKey(W)
                pass

            elif action_index is 1:

                ReleaseKey(W)
                ReleaseKey(D)
                ReleaseKey(S)

                PressKey(A)

                if total_time % 8 == 0:
                    PressKey(W)


            elif action_index is 2:

                ReleaseKey(W)
                ReleaseKey(A)
                ReleaseKey(S)

                PressKey(D)
                if total_time % 8 == 0:
                    PressKey(W)

            elif action_index is 3:
                ReleaseKey(W)
                ReleaseKey(A)
                ReleaseKey(D)

                PressKey(S)
                if total_time % 8 == 0:
                    PressKey(W)

            elif action_index is 4:
                ReleaseKey(W)
                ReleaseKey(A)
                ReleaseKey(D)
                ReleaseKey(S)
                if total_time % 8 == 0:
                    PressKey(W)

        print("Total Time: ", total_time)
        print("Policy Vector: ", q_values)
        print("Action: ", action_index)

        total_time += 1
