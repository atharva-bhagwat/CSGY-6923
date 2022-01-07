# Function to Load Dataset
load_dataset = function(csvfile){read.csv(csvfile, stringsAsFactors=TRUE)}

# Pretty Print Function
pprint <- function(text, variable){
	cat('\n')
	cat(text)
	if(!missing(variable)){
		cat('\n')
		print(variable)
	}
}