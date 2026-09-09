from pydantic import BaseModel,ValidationError,EmailStr
import json

class USERINPUT(BaseModel):#继承类
    name: str
    email: EmailStr
    query: str

#creat a userinput instance

user_input = USERINPUT(
    name = "Yuanhome",
    email = "52emancipationfoever@gmail.com",
    query = "how about staying together forever?"
)
print(user_input)
