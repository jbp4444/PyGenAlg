
import csv

def main():
	zip_to_congr = {}
	with open('zip_to_congr.csv','r') as fp:
		readCSV = csv.reader( fp, delimiter=',' )
		# skip 2 headers
		next( readCSV, None )
		next( readCSV, None )
		# now read all the actual data
		for row in readCSV:
			zip_to_congr[row[1]] = 1

	nc_zips = {}
	with open('nc_data.zip.txt','r') as fp:
		readCSV = csv.reader( fp, delimiter=',' )
		# skip 2 headers
		next( readCSV, None )
		next( readCSV, None )
		# now read all the actual data
		for row in readCSV:
			nc_zips[row[1]] = 1

	all_zips = {}
	with open('gaz2015zcta5centroid.csv','r') as fp:
		readCSV = csv.reader( fp, delimiter=',' )
		# skip 1 header
		next( readCSV, None )
		# now read all the actual data
		for row in readCSV:
			all_zips[row[2]] = 1

	nc_zips2 = {}
	with open('pop_per_zip.csv','r') as fp:
		readCSV = csv.reader( fp, delimiter=',' )
		# skip 2 headers
		next( readCSV, None )
		next( readCSV, None )
		# now read all the actual data
		for row in readCSV:
			nc_zips2[row[1]] = 1

	print( 'scanning nc_data.zip.txt to zip_to_congr.csv ...' )
	for zip in nc_zips.keys():
		if( not zip in zip_to_congr ):
			print( 'NOT found: '+zip )

	print( '\nscanning zip_to_congr.csv to nc_data.zip.txt ...' )
	for zip in zip_to_congr.keys():
		if( not zip in nc_zips ):
			print( 'NOT found: '+zip )


	print( '\nscanning nc_data.zip.txt to all_zips ...' )
	for zip in nc_zips.keys():
		if( not zip in all_zips ):
			print( 'NOT found: '+zip )

	print( '\nscanning zip_to_congr.csv to all_zips ...' )
	for zip in zip_to_congr.keys():
		if( not zip in all_zips ):
			print( 'NOT found: '+zip )

	print( '\nscanning pop_per_zip to all_zips ...' )
	for zip in nc_zips2.keys():
		if( not zip in all_zips ):
			print( 'NOT found: '+zip )

	print( '\nscanning pop_per_zip to zip_to_congr.csv ...' )
	for zip in nc_zips2.keys():
		if( not zip in zip_to_congr ):
			print( 'NOT found: '+zip )

if __name__ == '__main__':
	main()
