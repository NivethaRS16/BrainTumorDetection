<?php
 $db=mysqli_connect("localhost","id16188360_projects","Iamedotembedded@1","id16188360_project")or die(mysqli_connect_error());
if(isset($_GET['back']))
{
    echo ' <script language="javascript" type="text/javascript">
parent.document.location="home.php";
</script>';    
    
}
$query=mysqli_query($db,"SELECT `id` FROM `patient_login` WHERE 1 ") or die(mysqli_error($db));
$row1=mysqli_fetch_array($query);
$id=$row1['id'];

if($id!=NULL)
{
$query=mysqli_query($db,"SELECT * FROM `patient_details` WHERE `id`=$id ") or die(mysqli_error($db));
$row=mysqli_fetch_array($query);
$name=$row['name'];
$age=$row['age'];
$dob=$row['dob'];
$address=$row['address'];
$email=$row['email'];
}

?>


<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
.card {
  box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
  transition: 0.3s;
  width: 80%;
  border-radius: 5px;
}

.card:hover {
  box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}

img {
  border-radius: 5px 5px 0 0;
}

.container {
  padding: 2px 16px;
  font-size="10%";
}
</style>
</head>
<body>

<center><h2>Details</h2></center>
<center>

<form method="GET" action="details.php">
<div class="card">
  <img src="pd.png" alt="Avatar" style="width:100%">
  <div class="container">
    <h4><b><?php echo "$name" ?></b></h4> 
    <p><?php echo "$dob" ?></p> 
    <h4><b><?php echo "$age" ?></b></h4> 
    <p><?php echo "$address" ?></p> 
    <p><?php echo "$email" ?></p> 
    <br><br>
   <button type="submit" name="back" class="registerbtn">Back</button>
 <br><br>
  </div>
</div>
</form>
</center>
<br><br>
</body>
</html> 
