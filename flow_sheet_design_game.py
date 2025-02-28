 
import numpy as np
import copy
from typing import Tuple, List, Dict
from gaz_singleplayer.config_syngame import Config
from environment.flowsheet_simulation import FlowsheetSimulation
from environment.env_config import EnvConfig
import pprint
 
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
 
class SinglePlayerGame:
    """
    Class representing a single-player task for flowsheet synthesis.
    """
 
    def __init__(self, config: Config, flowsheet_simulation_config: EnvConfig, problem_instance: List):
        """
        Initialize the game.
 
        Parameters
        ----------
        config : Config
            Configuration for the game.
        flowsheet_simulation_config : EnvConfig
            Environment-specific parameters and settings.
        problem_instance : List
            List of feed streams representing the problem instance.
        """
        self.config = config
        self.flowsheet_simulation_config = flowsheet_simulation_config
        self.problem_instance = problem_instance
 
        self.level_current_player = 0
        self.action_current_player = {"line_index": None, "unit_index": None, "spec_cont": None, "spec_disc": None}
        self.game_broken = False
       
        self.player_environment = FlowsheetSimulation(
            copy.deepcopy(self.problem_instance),
            self.flowsheet_simulation_config
        )
 
        self.current_feasible_actions = self.player_environment.get_feasible_actions(
            current_level=self.level_current_player,
            chosen_stream=self.action_current_player["line_index"],
            chosen_unit=self.action_current_player["unit_index"]
        )
 
        self.game_is_over = False
        self.player_npv = -float("inf")
        self.player_npv_explicit = -float("inf")
 
    def get_current_level(self):
        return self.level_current_player
 
    def get_objective(self, for_player: int) -> float:
        return self.player_npv
 
    def get_explicit_npv(self, for_player: int) -> float:
        return self.player_npv_explicit
 
    def get_sequence(self, for_player: int) -> Dict:
        return self.player_environment.blueprint
 
    def get_number_of_lvl_zero_moves(self) -> int:
        return self.player_environment.steps
 
    def get_num_actions(self) -> int:
        """
        Legal actions for the current player at the current level given as a list of ints.
        """
        return len(self.current_feasible_actions)
 
    def get_feasible_actions_ohe_vector(self) -> np.array:
        return self.current_feasible_actions
 
    def is_finished_and_winner(self) -> Tuple[bool, int]:
        # Irrelevant for single-player games
        return self.game_is_over, 0
 
    def make_move(self, action: int) -> Tuple[bool, float, bool]:
        """
        Performs a move in the game environment. In the flowsheet case, this does not necessarily mean
        that a unit is placed, as the action may not be complete.
 
        Parameters
        ----------
        action : int
            The index of the action to play. The action index should be feasible.
 
        Returns
        -------
        game_done : bool
            Whether the flowsheet is finished.
        reward : float
            Reward based on the NPV.
        move_worked : bool
            Whether the move led to a converging flowsheet.
        """
        if self.game_broken:
            raise Exception('Playing in a broken game')
 
        if self.current_feasible_actions[action] != 1:
            raise Exception("Playing infeasible action.")
 
        # Update the current action state based on the game level
        if self.level_current_player == 0:
            self.action_current_player["line_index"] = action
        elif self.level_current_player == 1:
            self.action_current_player["unit_index"] = action
        elif self.level_current_player == 2:
            self.action_current_player["spec_disc"] = action
        elif self.level_current_player == 3:
            self.action_current_player["spec_cont"] = [None, [action, None]]
 
        # Determine the next level based on the current level and action
        next_level = self._determine_next_level(action)
 
        # Execute the action in the simulation if the next level is 0
        move_worked = True
        game_done = False
        reward = 0.0
 
        if next_level == 0:
            game_done, reward, move_worked = self._execute_action()
 
        # Update the level and feasible actions for the next move
        if not self.game_is_over:
            self.level_current_player = next_level
            if next_level == 0:
                self.action_current_player = {"line_index": None, "unit_index": None, "spec_cont": None, "spec_disc": None}
            if not self.game_broken:
                self.current_feasible_actions = self.player_environment.get_feasible_actions(
                    current_level=self.level_current_player,
                    chosen_stream=self.action_current_player["line_index"],
                    chosen_unit=self.action_current_player["unit_index"]
                )
 
        return game_done, reward, move_worked
 
    def _determine_next_level(self, action: int) -> int:
        """Determine the next level based on the current game state and action."""
        if self.level_current_player == 0:
            return 0 if action == len(self.current_feasible_actions) - 1 else 1
        elif self.level_current_player == 1:
            return self.flowsheet_simulation_config.unit_types[
                self.flowsheet_simulation_config.units_map_indices_type[action]
            ]["next_level"]
        elif self.level_current_player in [2, 4]:
            return 0
        elif self.level_current_player == 3:
            return 4
 
    def _execute_action(self) -> Tuple[bool, float, bool]:
        """Execute the current action in the simulation."""
        _, npv, npv_normed, flowsheet_synthesis_complete, convergent = self.player_environment.place_apparatus(
            line_index=self.action_current_player["line_index"],
            apparatus_type_index=self.action_current_player["unit_index"],
            specification_continuous=self.action_current_player["spec_cont"],
            specification_discrete=self.action_current_player["spec_disc"]
        )
 
        if convergent:
            if flowsheet_synthesis_complete:
                self.player_npv_explicit = npv
                self.player_npv = npv if not self.player_environment.config.norm_npv else npv_normed
                self.game_is_over = True
                return True, self.player_npv, True
        else:
            self.game_broken = True
            return False, 0.0, False
 
    def get_current_state(self):
        """Returns the current game state as a dictionary."""
        if self.game_broken:
            raise Exception("Getting state of a broken game.")
       
   
 
        state = {
            "current_npv": self.player_environment.current_net_present_value * self.config.objective_scaling,
            "num_lines": len(self.player_environment.state_simulation["list_line_information"]),
            "action_level": self.level_current_player,
            "feasible_actions": self.current_feasible_actions,
            "flowsheet_finished": self.player_environment.state_simulation["flowsheet_syn_done"],
            "chosen_stream": self.action_current_player["line_index"],
            "chosen_unit": self._get_one_hot_encoded_unit(),
        }
        return state
 
    def _get_one_hot_encoded_unit(self) -> np.array:
        one_hot = np.zeros(self.flowsheet_simulation_config.num_units)
        if self.action_current_player["unit_index"] is not None:
            one_hot[self.action_current_player["unit_index"]] = 1
        return one_hot
 
    def copy(self):
        return copy.deepcopy(self)
 
    @staticmethod
    def generate_random_instance(flowsheet_simulation_config) -> Dict:
        return flowsheet_simulation_config.create_random_problem_instance()
 
# Interactive Shell, where the game starts
def play_game(game: SinglePlayerGame):
    print("Welcome to the Single-Player Flowsheet Synthesis Task!")
    print("Type 'helps' for instructions or 'quit' to exit.")
 
    while not game.game_is_over: # Game Loop until the game is over
        state = game.get_current_state() # Get the current state of the game
        print("\nCurrent State:")
        print(f" - Current NPV: {state['current_npv']}")
        print(f" - Feasible Actions: {np.nonzero(state['feasible_actions'])[0]}")
        print(f" - Current Level: {state['action_level']}")
 
        action = input("Enter your action (index) or 'quit': ").strip()
 
        if action.lower() == 'quit':
            print("Exiting the game. Goodbye!")
            break
 
        if action.lower() == 'help':
            print("Commands:")
            print(" - Enter the index of a feasible action to make a move.")
            print(" - Type 'quit' to exit.")
            continue
 
        try:
            action = int(action) # the action made is converted to an integer, the action is then passed to make_move function
                                 #  which returns the game_done, reward and move_worked
            game_done, reward, move_worked = game.make_move(action)
 
            if game_done:
                print(f"Game Over! Final NPV: {reward}") # If the game is over, print the final NPV
                break
        except ValueError:
            print("Invalid input. Please enter a valid action index or 'quit'.")
        except Exception as e:
            print(f"Error: {e}")
 
# Main Function
def main():
    # Replace these with actual implementations of Config and EnvConfig
    config = Config()
    env_config = EnvConfig()
    problem_instance = env_config.create_random_problem_instance()
 
    game = SinglePlayerGame(config, env_config, problem_instance)
    play_game(game)
 
if __name__ == "__main__":

    main()
 
 
