# Hearthstone Archetype Prediction

This project is used to train recurrent neural networks on data extracted from Hearthstone game logs in order to predict the archetype used by the players given cards played on the first turn or the first few turns. 

You will need [Confusion Matrix Pretty Print](https://github.com/wcipriano/pretty-print-confusion-matrix), pandas, seaborn, pytorch, hsreplay, hslog, hearthstone and their dependencies (numpy and matplotlib being the big ones).

You will also need hearthstone replays, in the xml format understood by the hearthstone python library **and** (for training and calculation of the metrics) ground truth data on the archetypes used in the games. Archetypes have to provided in a json file that contains a list of dictionaries with keys "id" (= name of the replay file without extension), "player1_archetype" (= ID of the first player's deck archetype) and "player2_archetype" (= ID of the second player's deck archetype). Additionally, and archetypes.json file that maps archetype IDs to names should be provided for nicer formatting.

## Data Preprocessing

`analyze_log.py` will read a single replay file, or a directory of replay files and extract the data needed for prediction (cards played, turn numbers, time taken by the players), and store them in csv-files. Each line in the file represents the actions taken by a single player during a single game, meaning there will be two lines per game, one for each player. Each row starts with the game id and player id, and continues with triples of card id, time since last action and turn number, for up to 11 actions (we were only interested in the first turn taken by a player, this may have to be extended when more turns should be taken into account). The last two columns of each line contain the ground truth archetype ID and name.


`check_replay.py` is a small utility to verify that a single replay file or a directory of replay files is correctly readable by the library and split them up into training, validation and test sets. A typical workflow consists in using this tool to split the data into three sets, and then applying `analyze_log.py` on each of these directories to produce three csv files for use with `learn_archetypes.py` (described below).

## Training

`learn_archetypes.py` uses the csv-files produced by the preprocessing described above. It will train three different classifiers on the training set:
 - A simple baseline-classifier that just predicts the most common deck type for each class
 - A Bayesian classifier that estimates prior probabilities from the observed cards and predicts the deck type by calculating estimated posterior probabilities and predicting the highest probability class 
 - A classifier using a Recurrent Neural Network that is passed the played cards as a sequence and produces an archetype prediction as the output. For this classifier, cards are encoded using a one-hot-encoding before being passed to the network.
 
The script will train these classifiers on the training set, and supports hyper-parameter tuning on the validation set. There is also an option to calculate the performance metrics on the given test set. 