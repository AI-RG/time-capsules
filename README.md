# A Simple Capsule Network

In [1], the concept of a "capsule" was introduced. The idea is to represent features using vectors; then, the norm of a given vector encodes the probability that its feature exists. Each such vector was dubbed a *capsule* in [1]. This extra structure permits information to be routed between capsules of successive layers in a more nuanced way than is possible between simple nodes in vanilla MLPs or CNNs. Namely, the effective weight between capsule *i* of layer *L* and capsule *j* of layer *L+1* is influenced by the dot product *v<sub>i</sub>.v<sub>j</sub>*. ([1] presents the full details of this routing process quite clearly, so I won't reproduce them here.)

[1] used an architecture as in the figure to classify digits in MNIST and multi-MNIST, single images of which contain two (overlapping) digits.

<img src="https://github.com/AI-RG/baselines/blob/master/baselines/a2c2/capsule-policy-original.png" alt="orig-caps-policy" width="800px"/>

## Experiments

<img src="https://github.com/AI-RG/time-capsules/blob/master/caps/capsule-accurcy.png" alt="caps-test-acc" width="600px"/>

We train on MNIST and multi-MNIST, yielding similar performance to [1], namely 99.5% test set accuracy on MNIST and 77.8% test set accuracy on multiMNIST.

## Extensions

### Reinforcement Learning

We also investigate CapsNets in a reinforcement learning setting; the repository is https://github.com/AI-RG/baselines/tree/master/baselines/a2c2/capsules. In this setting, a CapsuleNet was appealing for a few reasons, but I was most interested in using capsules to represent actions in a discrete policy.

The more modular organization of casule networks lends them to transfer learning [2]. As such, an interesting experiment in this setting would be to test how well a pretrained (e.g. Reptile [4]) CapsuleNet could adapt to different environments, e.g. different Atari games. This is the subject of ongoing work. 

### Recurrence

There is already an element of recurrence in CapsNets, namely the dynamic routing procedure (see algorithm 1 in [1]), which iteratively estimates how much weight *c<sub>ij</sub>* to assign capsule *j* of layer *L* in determining the value of capsule *i* of layer *L+1*. The most straightforward way to add recurrence to CapNets is simply to refrain from resetting *c<sub>ij</sub>*  (or more properly *b<sub>ij</sub>*) before each new observation; this amounts to adding a prior of the previous observations' routings to the current one.

The addition of a GRU-like gated update mechanism would represent a larger modification in the direction of recurrence. Using the final layer ("Encoded Caps") as the hidden state, one could form a gated update with new input *x' = PrimaryCaps(x)* the outupt of PrimaryCaps. Since the `squash()` nonlinearity normalizes inputs, it is already well suited for use in a recurrent architecture.

Experiments in this direction are ongoing. 

## Details

### How to run this code

The command `python3 capsule_trainer.py` runs the 1M batches (size 128) on MNIST. Additional command line options available; for a listing, use help (`-h`).

## Bibliography

- [1] S. Sabour, N. Front, and G. Hinton, "Dynamic Routing Between Capsules" (arXiv: 1710.09829)
- [2] A. Gritsevskiy and M. Korablyov, "Capsule networks for low-data transfer learning" (arXiv: 1804.10172) 
- [3] V. Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (arXiv:1602.01783)
- [4] A. Nichol and J. Shulman, "Reptile: A Scalable Metalearning Algorithm" (arXiv: 1803.02999)
