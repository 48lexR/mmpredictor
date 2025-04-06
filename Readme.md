# Welcome to 48lexR's March Madness Neural Network Predictor.

I reserve all rights as the sole contributor to the design of this model and its code. I attest that Ken Pomeroy's NCAA rankings was used in this model's training. All information used in the training of this model was attained through publicly available means.

## Model Design

This model is based on a simple mathematical concept: Ken Pomeroy's NCAA rankings for each team can be boiled down into a single 4-D vector consisting of the following components:

1. a team's NET rating,
2. a team's OFF rating,
3. a team's DEF rating,
4. a team's ADJ rating.

The goal of this design, then, is to map the plane formed by two teams' ratings onto a 2D vector representing the score.

Currently, this model's loss is estimated at 17.8 on average evaluation. This loss is calculated by measuring the distance from the predicted output vector to the actual result.
