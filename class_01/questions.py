"""def extendList(val, mlist=None):
	if mlist == None:
		mlist = []
	mlist.append(val)
	return mlist

list1 = extendList(10)
list2 = extendList(123,[])
list3 = extendList('a')

print "list1 = %s" % list1
print "list2 = %s" % list2
print "list3 = %s" % list3"""

class A():
    x = 1
    
a = A()
print a.x

b = [A() for ix in range(5)]

for ix in b:
    print ix

b[2].x = 5

for ix in b:
    print ix.x


import copy
q = copy.deepcopy(b)	# recursive-deep clone
p = copy.copy(b)	# shallow clone

try:
	print '1' + 'a'
except:
	print 'got an error'


















