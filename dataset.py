import urllib.request

url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
file_path = "dataset/insurance.csv"

# Download the file
urllib.request.urlretrieve(url, file_path)

print("Dataset downloaded successfully!")
