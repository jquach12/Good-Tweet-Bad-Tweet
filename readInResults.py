import sys
#used this function to get a more accurate accurate score as i did not know my training data contained NO neutral sentiment


def f1Score(resultsFileName):


	with open(resultsFileName) as f:

		tp = 0
		fp = 0

		tn = 0
		fn = 0

		truePos = "4 DOES MATCH THE LABEL: 4"
		falsePos = "4 DOESNT MATCH THE LABEL: 0"

		trueNeg = "0 DOES MATCH THE LABEL: 0"
		falseNeg = "0 DOESNT MATCH THE LABEL: 4"

		content = f.readlines()
		for line in content:
			

			if line.startswith(truePos):
				tp += 1
			elif line.startswith(falsePos):
				fp += 1
			elif line.startswith(trueNeg):
				tn += 1
			elif line.startswith(falseNeg):
				fn += 1

	precision = float(tp) / (tp + fp) #we gotta float them in order to avoid integer division (ESPECIALLY WITH 0.X vs 0)

	recall = float(tp) / (tp + fn)

	f1Score = 2*((precision*recall) / (precision + recall))
	print("Precision: %s" %precision)
	print("Recall: %s" %recall)
	print("F1Score: %s" %f1Score)
	return f1Score

if __name__ == '__main__':
	fileName = sys.argv[1]
	f1Score(fileName)
