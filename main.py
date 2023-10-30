from fastapi import Body, FastAPI
from fastapi.encoders import jsonable_encoder

app = FastAPI()

# import motor.motor_asyncio
# client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
# database = client.students
# student_collection = database.get_collection("students_collection")

from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
database = client.students
student_collection = database.get_collection("students_collection")

def student_helper(student):
    print(student["fullname"])
    return {
        "id": str(student["_id"]),
        "fullname": student["fullname"],
        "course_of_study": student["course_of_study"],
        "year": student["year"],
        "GPA": student["gpa"],
    }

# Retrieve all students present in the database
async def retrieve_students():
    students = []
    for student in student_collection.find(): # type: ignore
        students.append(student_helper(student))
    return students


# Add a new student into to the database
async def add_student(student_data: dict):
    student =  student_collection.insert_one(student_data) # type: ignore
    new_student =  student_collection.find_one({"_id": student.inserted_id}) # type: ignore
    return student_helper(new_student)

@app.get("/")
async def root(): # type: ignore 
    students = []
    for student in student_collection.find(): # type: ignore
        students.append(student_helper(student))
    print(students)
    return students

    
from pydantic import BaseModel, Field
class StudentSchema(BaseModel):
    fullname: str 
    course_of_study: str 
    year: int 
    gpa: float 

    class Config:
        json_schema_extra = {
            "example": {
                "fullname": "John Doe",
                "course_of_study": "Water resources engineering",
                "year": 2,
                "gpa": "3.0",
            }
        }

def ResponseModel(data, message):
    return {
        "data": [data],
        "code": 200,
        "message": message,
    }

@app.post("/")
async def add_student_data(student: StudentSchema = Body(...)):
    student = jsonable_encoder(student)
    new_student = await add_student(student) # type: ignore
    return ResponseModel(new_student, "Student added successfully.")
