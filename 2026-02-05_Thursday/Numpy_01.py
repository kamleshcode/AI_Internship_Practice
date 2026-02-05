import sys
import numpy as np
import time

# ======================Creating Array=========================
# arr_1d = np.array([1, 2, 3, 4, 5])
# arr_2d = np.array([[1,2,3], [4,5,6]])
# print(f'Array 1:{arr_1d}')
# print(f'Array 2:{arr_2d}')
#=========================Numpy Array v/s list =======================
py_list = [1,2,3]
print(f'Multiplying list:{py_list *2}') #[1, 2, 3, 1, 2, 3]
py_list2=np.array([1,2,3])
print(py_list2 *2) # It multiply every element by 2 : [2 4 6]

start = time.time()
py_list3=[ i*2 for i in range(1000)]
print(f'Time taken for multiplying list:{time.time() - start}')
print(f'Memory Size of list:{sys.getsizeof(py_list3)}')
start =time.time()
py_list4=np.arange(10)*2
print(f'Numpy taken for multiplying list:{time.time() - start}')
print(f'Memory Size of list:{sys.getsizeof(py_list4)}')

# =======================Creating Array==========================
# zeros = np.zeros((2, 3))
# print(f'Zeros matrix:{zeros}')
# ones = np.ones((2, 3))
# print(f'Ones matrix:{ones}')
# full =np.full((2, 3),5)
# print(f'Full matrix with value 5:{full}')
# random =np.random.random((2, 3)) #random is a class which have random method
# print(f'Random matrix:{random}')
# sequence=np.arange(0,9)
# print(f'Sequence matrix:{sequence}')

# ========================Vector,Matrix and tensor======================
# #Vector: 1d array in NumPy,Matrix:2d array,Tensors: any array with more than 2 dimensions
# vector =np.array([1,2,3])
# print(f'Vector matrix:{vector}')
# matrix=np.array([[1,2,3],
#                  [4,5,6]])
# print(f'Matrix matrix:{matrix}')
# tensor=np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
# print(f'Tensor matrix:{tensor}')
# print(f'Tensor shape:{tensor.shape}')

#=======================Array Properties==========================
# arr1=np.array([[1,2,3],
#                  [4,5,6.0]])
# print(f'Array 1:{arr1}')
# print(f'Shape:{arr1.shape}')
# print(f'Dimension:{arr1.ndim}')
# print(f'Size:{arr1.size}')
# print(f'Dtype:{arr1.dtype}')

# =========================Array Reshaping==========================
# arr2 = np.arange(12)
# reshaped=arr2.reshape(3,4)
# print(f'Reshaped :{reshaped}')
# flattened=reshaped.flatten()
# print(f'Flattened :{flattened}')
# raveled=reshaped.ravel() #returns view instead of copy
# print(f'Raveled :{raveled}')
# transposed=reshaped.T
# print(f'Transposed :{transposed}')

#=======================Slicing========================
# arr1=np.array([1,2,3,4,7,5])
# print(f'Basic Slicing:{arr1[1:2]}')
# print(f'With step:{arr1[::2]}')
# print(f'Negative step:{arr1[::-1]}')
#
# #SLicng in 2d array : array[row_slice, column_slice]
# arr2=np.array([[1,2,3],
#                [3,4,6],
#                [7,8,9]])
# print(f'Specific elements:{arr2[0:2,:2]}')
# print(f'Entire row:{arr2[1:3]}')
# print(f'Entire column:{arr2[:,1:3]}')

#=====================Sorting========================
# arr8=np.array([1,4,8,9,0,34,15,6,7,2,3])
# print(f'Sorted Array:{np.sort(arr8)}')
# # 2d array sorting
# arr9=np.array([[1,2,3],
#                [2,3,1],
#                [2,1,3]])
# print("Sorted 2D Array column:",np.sort(arr9, axis=0))
# print("Sorted 2D Array rows:",np.sort(arr9, axis=1))

#======================Filter=======================
# numbers=np.array([1,2,3,4,5,6,7,8,9])
# print(f'Numbers:{numbers}')
# even_numbers=numbers[numbers % 2 == 0]
# odd_numbers=numbers[numbers % 2 != 0]
# print(f'Event Numbers: {even_numbers}')
# print(f'Odd Numbers: {odd_numbers}')
# # filter with mask
# mask=numbers > 5
# print(f'Mask: {mask}')
# print(f'Masked Array:{numbers[mask]}')

#fancy indexing vs np.where
# indices=[0,2,5]
# print(numbers[indices])
# where_result1 = np.where(numbers > 7)
# where_result2 = np.where(numbers > 7, True, False)
# print(where_result1)
# print(where_result2)

#===============Adding, removing & deleting data================
# arr1=np.arange(10)
# arr2=np.arange(5)
# combined=np.concatenate((arr1,arr2)) # note:concatenate(()) tuple is passed
# print(combined)

# original=np.array([[1,2],[2,5],[8,9]])
# arr2=np.array([[11,12]])
# add_new_row=np.vstack((original,arr2))
# print(f'Original: {original}')
# print(f'Added new row: {add_new_row}')
# arr3=np.array([[13],[15],[16]])
# add_new_column=np.hstack((original,arr3))
# print(f'Added new column: {add_new_column}')
# arr4=np.array([1,2,3,4,5])
# delete_element=np.delete(arr4,2,axis=0)
# print(f'Deleted element: {delete_element}') #when you print it returns array updated array not deleted elements

#===================Array Compatibility==========================
# arr1=np.arange(10)
# arr2=np.arange(5)
# print(f'Compatibility : {arr1.shape == arr2.shape}')








