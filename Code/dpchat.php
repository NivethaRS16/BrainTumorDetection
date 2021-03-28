<?php
 $db=mysqli_connect("localhost","id16188360_projects","Iamedotembedded@1","id16188360_project")or die(mysqli_connect_error());



if(isset($_GET['back']))
{
echo ' <script language="javascript" type="text/javascript">
parent.document.location="pchat.php";
</script>';

}
if(isset($_GET['send']))
{
if(isset($_GET['msg']))
{
    $msg=$_GET['msg'];
           $query=mysqli_query($db,"SELECT * FROM `doctor_login` WHERE 1 ") or die(mysqli_error($db));
             $row2=mysqli_fetch_array($query);
		        $id=$row2['id'];
	         $query=mysqli_query($db,"SELECT * FROM `patient_chat_id` WHERE 1 ") or die(mysqli_error($db));
             $row1=mysqli_fetch_array($query);
		        $did=$row1['id'];
	        
	        $query1=mysqli_query($db, "INSERT INTO `chat` (`user_id`, `doctor_id`, `mes`, `ft`) VALUES ('$did','$id','$msg',2)") or die(mysqli_error($db));
 echo ' <script language="javascript" type="text/javascript">
parent.document.location="dpchat.php";
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
  margin: 0 auto;
  max-width: 800px;
  padding: 0 20px;
}

/* Add padding to containers */
.container2 {
  padding: 16px;
  background-color: white;
}

/* Full-width input fields */
input[type=text], input[type=password] {
  width: 90%;
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

.container {
  border: 2px solid #dedede;
  background-color: #f1f1f1;
  border-radius: 5px;
  padding: 10px;
  margin: 10px 0;
}

.darker {
  border-color: #ccc;
  background-color: #ddd;
}

.container::after {
  content: "";
  clear: both;
  display: table;
}

.container img {
  float: left;
  max-width: 60px;
  width: 100%;
  margin-right: 20px;
  border-radius: 50%;
}

.container img.right {
  float: right;
  margin-left: 20px;
  margin-right:0;
}

.time-right {
  float: right;
  color: #aaa;
}

.time-left {
  float: left;
  color: #999;
}
</style>
</head>
<body>
<form method="GET" action="dpchat.php">
<button type="submit" name="back" class="registerbtn" >Back</button>
</form>
<h2>Chat Messages</h2>


                <?php  
             $query=mysqli_query($db,"SELECT * FROM `doctor_login` WHERE 1 ") or die(mysqli_error($db));
             $row=mysqli_fetch_array($query);
		        $id=$row['id'];
	         $query=mysqli_query($db,"SELECT * FROM `patient_chat_id` WHERE 1 ") or die(mysqli_error($db));
             $row1=mysqli_fetch_array($query);
		        $did=$row1['id'];
	    
                

				$query=mysqli_query($db,"SELECT * FROM `chat` WHERE `doctor_id`= '$id' && `user_id`= '$did'") or die(mysqli_error($db));
                $x=1;
while($row=mysqli_fetch_array($query))
{   
     $user_id=$row['user_id'];
	$message=$row['mes'];
	$ft=$row['ft'];
	$dt=$row['dt'];
	  if($ft==1)
	  {
echo "<div class=\"container darker\">
  <img src=\"pat.png\" alt=\"Avatar\" style=\"width:100%;\">
  <p>$message</p>
  <span class=\"time-right\">$dt</span>
</div><br><br>

";  
}
else
{
    
    echo " 
<div class=\"container\">
  <img src=\"doc.jpg\" alt=\"Avatar\" class=\"right\" style=\"width:100%;\">
  <p>$message</p>
  <span class=\"time-left\">$dt</span>
<br><br>
";
}
}
?> 
<br><br>

<form method="GET" action="dpchat.php">
    <label for="mes"><b>Send message</b></label>
    <input type="text" placeholder="Enter Message" name="msg"  required>
    <button type="submit" name="send" class="registerbtn">Send</button>


</form>
</div>

</body>
</html>
