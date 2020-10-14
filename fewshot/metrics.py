def simple_accuracy(ground_truth, predictions):
  score = 0
  i = 0
  indexes = []
  for truth, pred in zip(ground_truth, predictions):
    i += 1
    if truth == pred:
      indexes.append(i)
      score += 1

  #print(score / len(ground_truth) * 100)
  return score / len(ground_truth) * 100

def simple_topk_accuracy(ground_truth, predictions):
  score = 0
  i = 0
  indexes = []
  for truth, topk in zip(ground_truth, predictions):
    i += 1
    if truth in topk:
      indexes.append(i)
      score += 1

  #print(score / len(ground_truth) * 100)
  return score / len(ground_truth) * 100 