# CTC-based-alignment
pytorch ctc-based alignment implementation
(akin to Viterbi forced alignment)

```
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
S = 15      # Target sequence length

log_probs = torch.randn(T, C).log_softmax(1)
targets = torch.randint(low=1, high=C, size=(S,), dtype=torch.long)
results = ctc_alignment(log_probs, targets)
```