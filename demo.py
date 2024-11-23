import turtle

# Set up the screen
screen = turtle.Screen()
screen.title("Human Face Drawing")
screen.bgcolor("white")

# Set up the turtle
pen = turtle.Turtle()
pen.speed(5)

# Draw the face (circle)
pen.penup()
pen.goto(0, -200)
pen.pendown()
pen.circle(200)

# Draw the left eye (circle)
pen.penup()
pen.goto(-70, 50)
pen.pendown()
pen.circle(30)

# Draw the right eye (circle)
pen.penup()
pen.goto(70, 50)
pen.pendown()
pen.circle(30)

# Draw the nose (triangle)
pen.penup()
pen.goto(0, 50)
pen.pendown()
pen.goto(-30, -20)
pen.goto(30, -20)
pen.goto(0, 50)

# Draw the mouth (arc)
pen.penup()
pen.goto(-100, -50)
pen.setheading(-60)
pen.pendown()
pen.circle(100, 120)  # Draw an arc for the smile

# Draw the left ear
pen.penup()
pen.goto(-200, 50)
pen.pendown()
pen.circle(50)

# Draw the right ear
pen.penup()
pen.goto(150, 50)
pen.pendown()
pen.circle(50)

# Finish up
pen.hideturtle()
screen.mainloop()
