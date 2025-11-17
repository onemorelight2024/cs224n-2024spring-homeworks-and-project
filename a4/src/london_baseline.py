# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils
import tqdm

def main():
    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    accuracy = 0.0
    correct = 0
    total = 0
    predictions=[]
    for line in tqdm.tqdm(open('birth_dev.tsv', encoding='utf-8')):
        predictions.append("London")
    total, correct = utils.evaluate_places('birth_dev.tsv', predictions)
    accuracy=correct/total
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
