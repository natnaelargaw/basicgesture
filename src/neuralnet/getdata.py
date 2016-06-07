import  cv2

ref = cv2.imread("/home/ati/Desktop/2.png")
gray=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
# print(gray)
data =[]

# print(len(gray))
# print(gray[0])
for i in range(len(gray)):
    tmp=gray[i]
    for j in range(len(tmp)):
        data.append(tmp[j])
        # print(tmp[j],)
print(data)
print(len(data))
# print(len(gray[0]))

# for i in gray:
#
#     print(i)
# print(w,h,c)
# print(x)
cv2.imshow("Letter", ref)


cv2.waitKey(0)
cv2.destroyAllWindows()