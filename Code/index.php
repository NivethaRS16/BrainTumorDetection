<?php
    $db=mysqli_connect("localhost","id16188360_projects","Iamedotembedded@1","id16188360_project")or die(mysqli_connect_error());

if(isset($_GET['sign']))
{
    
    
 if((isset($_GET['name']))&&(isset($_GET['dob']))&&(isset($_GET['age']))&&(isset($_GET['address']))&&(isset($_GET['email']))&&(isset($_GET['psw'])))
{

 $name=$_GET['name'];
 $dob=$_GET['dob'];
 $age=$_GET['age'];
 $address=$_GET['address'];
 $email=$_GET['email'];
 $psw=$_GET['psw'];
if(($name!=NULL)&&($dob!=NULL)&&($age!=NULL)&&($address!=NULL)&&($email!=NULL)&&($psw!=NULL))    
{
    
$query=mysqli_query($db,"SELECT `id` FROM `patient_details` WHERE `email`= '$email' ") or die(mysqli_error($db));
$row=mysqli_fetch_array($query);
$id=$row['id'];
if($id==NULL)
{
    $command="INSERT INTO `patient_details` (`name`, `dob`, `age`, `address`, `email`, `password`, `mri`) VALUES ('$name','$dob','$age','$address','$email','$psw', 'Nill')";

$query=mysqli_query($db, "INSERT INTO `patient_details` (`name`, `dob`, `age`, `address`, `email`, `password`, `mri`) VALUES ('$name','$dob','$age','$address','$email','$psw', \"Nill\")") or die(mysqli_error($db));

echo ' <script language="javascript" type="text/javascript">
alert("Registration Completed Successfully");
parent.document.location="login.php";
</script>';

}
else
{
    echo ' <script language="javascript" type="text/javascript">
alert("Email id already exists, please use another");
parent.document.location="index.php";
</script>';
}
}
else
{
    
    echo ' <script language="javascript" type="text/javascript">
alert("Please Fill All Fields");
parent.document.location="index.php";
</script>';
    
}
}

else
{
    
    echo ' <script language="javascript" type="text/javascript">
alert("Please Fill All Fields");
parent.document.location="index.php";
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
    <center><h1>Registration</h1></center>
      <br><center><img src="patient.png" alt="Nature" width="30%" height="30%">
</center></br>
    <p>Please fill in this form to create an account.</p>
    <hr>

<form method="GET" action="index.php">
    <label for="name"><b>Name</b></label>
    <input type="text" placeholder="Enter Name" name="name"  required>

    <label for="dob"><b>Date of Birth</b></label><br><br>
    <input type="date" placeholder="Enter DOB" name="dob"  required>
    <br>
     <br>
    <label for="age"><b>Age</b></label><br>
    <input type="number" placeholder="Enter Age" name="age"  required>
    <br><br>
    <label for="address"><b>Address</b></label>
    <input type="text" name="address"  required>

    <label for="email"><b>Email</b></label>
    <input type="text" placeholder="Enter Email" name="email" required>

    <label for="psw"><b>Password</b></label>
    <input type="password" placeholder="Enter Password" name="psw" required>

   <hr>
  
    <button type="submit" name="sign" class="registerbtn">Register</button>
    </form>
  </div>
  
  <div class="container signin">
    <p>Already have an account? <a href="login.php">Sign in</a>.</p>
  </div>


</body>
</html>
