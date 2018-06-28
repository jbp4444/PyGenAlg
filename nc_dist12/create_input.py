
import csv

# from 'check_files.py', the following were all in-sync
# with each other (no missing zip-code info):
#   scanning zip_to_congr.csv to all_zips ...
#   scanning pop_per_zip to all_zips ...
#   scanning pop_per_zip to zip_to_congr.csv ...

def main():
	# list of zip-codes (really ZCTA) in North Carolina
	# with population per zip
	pop_per_zip = {}
	with open('pop_per_zip.csv','r') as fp:
		readCSV = csv.reader( fp, delimiter=',' )
		# skip 2 headers
		next( readCSV, None )
		next( readCSV, None )
		# now read all the actual data
		for row in readCSV:
			pop_per_zip[row[1]] = row[3]

	# for each zip, what congressional district are they in now?
	zip_to_congr = {}
	with open('zip_to_congr.csv','r') as fp:
		readCSV = csv.reader( fp, delimiter=',' )
		# skip 2 headers
		next( readCSV, None )
		next( readCSV, None )
		# now read all the actual data
		for row in readCSV:
			zip_to_congr[row[1]] = row[2]

	# for each zip (in all US), what is the lat/long coords?
	zip_to_loc = {}
	with open('gaz2015zcta5centroid.csv','r') as fp:
		readCSV = csv.reader( fp, delimiter=',' )
		# skip 1 header
		next( readCSV, None )
		# now read all the actual data
		for row in readCSV:
			zip_to_loc[row[2]] = ( row[0],row[1] )

	total_pop = 0
	for zip in pop_per_zip.keys():
		pop    = pop_per_zip[ zip ]
		latlon = zip_to_loc[ zip ]
		congr  = str(int(zip_to_congr[ zip ]))
		data = [ zip, pop, latlon[1],latlon[0], congr ]
		print( ','.join(data ))

		total_pop = total_pop + int(pop)
	
	#print( 'total NC state pop = '+str(total_pop) )


if __name__ == '__main__':
	main()
