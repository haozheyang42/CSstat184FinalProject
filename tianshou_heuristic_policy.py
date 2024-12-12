from typing import Any, Dict, Optional, Union

import numpy as np
import random

from tianshou.data import Batch
from tianshou.policy import BasePolicy

class HeuristicPolicy(BasePolicy):
    """
    Always moves itself to maximize number of free tiles adjacent to self
    And removes tile to minimize number of free tiles adjacent to opponent
    """
    def __init__(self, board_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.board_size = board_size

    def _pos_in_board(self, pos):
        assert(len(pos) == 2)
        return all(0 <= i < j for i, j in zip(pos, self.board_size))

    def _encode_action(self, move_idx, remove_pos):
        board_area = self.board_size[0] * self.board_size[1]
        longer_side = max(self.board_size[0], self.board_size[1])
        return move_idx * board_area \
                    + remove_pos[0] * longer_side \
                    + remove_pos[1]
        
    def _decode_action(self, action):
        board_area = self.board_size[0] * self.board_size[1]
        longer_side = max(self.board_size[0], self.board_size[1])
        move_idx = action // board_area
        remainder = action % board_area
        remove_pos_0 = remainder // longer_side
        remove_pos_1 = remainder % longer_side
        return move_idx, (remove_pos_0, remove_pos_1)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute the random action over the given batch data.

        The input should contain a mask in batch.obs, with "True" to be
        available and "False" to be unavailable. For example,
        ``batch.obs.mask == np.array([[False, True, False]])`` means with batch
        size 1, action "1" is available but action "0" and "2" are unavailable.

        :return: A :class:`~tianshou.data.Batch` with "act" key, containing
            the random action.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        mask = batch.obs.mask
        obs = batch.obs.obs
        batch_size = len(mask)

        actions = []

        for i in range(batch_size):
            ep_mask = mask[i]
            ep_obs = obs[i]

            cur_pos = tuple(np.argwhere(ep_obs[:, :, 0] == 1)[0]) 
            opp_pos = tuple(np.argwhere(ep_obs[:, :, 1] == 1)[0]) 

            best_action = None
            best_score = 0
            
            for j in range(len(ep_mask)):
                if ep_mask[j]:
                    this_score = 0

                    move_idx, remove_pos = self._decode_action(j)

                    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    new_pos = (cur_pos[0] + MOVES[move_idx][0], cur_pos[1] + MOVES[move_idx][1])

                    pos1 = (new_pos[0] - 1, new_pos[1])
                    pos2 = (new_pos[0] + 1, new_pos[1])
                    pos3 = (new_pos[0], new_pos[1] - 1)
                    pos4 = (new_pos[0], new_pos[1] + 1)
                    
                    for p in (pos1, pos2, pos3, pos4):
                        # In board, present, not occupied by opponent
                        if self._pos_in_board(p) and ep_obs[:, :, 2][p] == 0 and ep_obs[:, :, 1][p] == 0:
                            this_score += 1

                    pos1 = (opp_pos[0] - 1, opp_pos[1])
                    pos2 = (opp_pos[0] + 1, opp_pos[1])
                    pos3 = (opp_pos[0], opp_pos[1] - 1)
                    pos4 = (opp_pos[0], opp_pos[1] + 1)
                    
                    if remove_pos in (pos1, pos2, pos3, pos4):
                        this_score += 1
                    
                    if this_score > best_score:
                        best_action = j
                        best_score = this_score

                    elif this_score == best_score:
                        best_action = random.choice([best_action, j])

            actions.append(best_action)

        return Batch(act=actions)


    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Since the heuristic agent learns nothing, it returns an empty dict."""
        return {}