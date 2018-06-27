
import unittest

from PyGenAlg import BaseChromo

class TestChromo(unittest.TestCase):
	def setUp(self):
		self.chromo = BaseChromo( size=4, dtype=int )
		self.chromo2 = BaseChromo( size=4, dtype=[int,int,float,float] )

	def test_size(self):
		self.assertEqual( self.chromo.chromo_sz, 4 )
	def test_size2(self):
		self.assertEqual( self.chromo.chromo_sz, len(self.chromo.data) )

	def test_dtype(self):
		self.assertListEqual( self.chromo.dataType, [int,int,int,int] )
	def test_dtype2(self):
		self.assertListEqual( self.chromo2.dataType, [int,int,float,float] )

	def test_pack(self):
		newdata = [1,2,3,4]
		self.chromo.setInitData( newdata )
		x = self.chromo.packData()
		self.chromo.unpackData( x )
		self.assertListEqual( self.chromo.data, newdata )

if __name__ == '__main__':
	unittest.main()
