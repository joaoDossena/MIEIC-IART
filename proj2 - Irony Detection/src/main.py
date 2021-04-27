import pandas as pd 

df = pd.read_csv("./datasets/train/train-taskA.txt", sep="	")

# print(df)

# tá lendo errado o dataframe. tem alguns que não tá lendo
# indices nao lidos (inclusive): 1646-1648, 3029-3040, 3459-3461
for i in range(1, len(df)):
	if(df['Tweet index'][i] != (df['Tweet index'][i-1] + 1)):
		print("Here")
		print("Current", df['Tweet index'][i])
		print("Previous", df['Tweet index'][i-1])


