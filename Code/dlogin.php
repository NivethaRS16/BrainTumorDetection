<?php
    $db=mysqli_connect("localhost","id16188360_projects","Iamedotembedded@1","id16188360_project")or die(mysqli_connect_error());

if(isset($_GET['login']))
{

    
    
 if((isset($_GET['email']))&&(isset($_GET['psw'])))
{
 $email=$_GET['email'];
 $psw=$_GET['psw'];
if(($email!=NULL)&&($psw!=NULL))    
{
    
$query=mysqli_query($db,"SELECT * FROM `doctor_details` WHERE `email`= '$email' && `password`= '$psw' ") or die(mysqli_error($db));
$row=mysqli_fetch_array($query);
$name=$row['name'];
$id=$row['id'];
if($name!=NULL)
{
    
    $query=mysqli_query($db,"UPDATE `doctor_login` SET `id`='$id' WHERE 1") or die(mysqli_error($db));

 echo ' <script language="javascript" type="text/javascript">
alert("Hello Dr. '.$name.' You are logged in");
parent.document.location="dhome.php";
</script>';


}
else
{
    echo ' <script language="javascript" type="text/javascript">
alert("Email id & Password , not matched or not found");
parent.document.location="dlogin.php";
</script>';
}
}
else
{
    
    echo ' <script language="javascript" type="text/javascript">
alert("Please Fill All Fields");
parent.document.location="dlogin.php";
</script>';
    
}
}

else
{
    
    echo ' <script language="javascript" type="text/javascript">
alert("Please Fill All Fields");
parent.document.location="dlogin.php";
</script>';
    
}   
   
}
?>


<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {
  font-family: Arial, Helvetica, sans-serif;
  background-color: white;
}

* {
  box-sizing: border-box;
}

/* Add padding to containers */
.container {
  padding: 16px;
  background-color: white;
}

/* Full-width input fields */
input[type=text], input[type=password] {
  width: 100%;
  padding: 15px;
  margin: 5px 0 22px 0;
  display: inline-block;
  border: none;
  background: #f1f1f1;
}

input[type=text]:focus, input[type=password]:focus {
  background-color: #ddd;
  outline: none;
}

/* Overwrite default styles of hr */
hr {
  border: 1px solid #f1f1f1;
  margin-bottom: 25px;
}

/* Set a style for the submit button */
.registerbtn {
  background-color: #4CAF50;
  color: white;
  padding: 16px 20px;
  margin: 8px 0;
  border: none;
  cursor: pointer;
  width: 100%;
  opacity: 0.9;
}

.registerbtn:hover {
  opacity: 1;
}

/* Add a blue text color to links */
a {
  color: dodgerblue;
}

/* Set a grey background color and center the text of the "sign in" section */
.signin {
  background-color: #f1f1f1;
  text-align: center;
}
</style>
</head>
<body>

  <div class="container">
    <center><h1>Login</h1></center>
      <br><center><img src="doctor.png" alt="Nature" width="10%" height="10%">
</center></br>

    <hr>

<form method="GET" action="dlogin.php">
    <label for="email"><b>Email</b></label>
    <input type="text" placeholder="Enter Email" name="email" required>

    <label for="psw"><b>Password</b></label>
    <input type="password" placeholder="Enter Password" name="psw" required>

   <hr>
  
    <button type="submit" name="login" class="registerbtn">Login</button>
    </form>
  </div>
  


</body>
</html>
