<?php
 $db=mysqli_connect("localhost","id16188360_projects","Iamedotembedded@1","id16188360_project")or die(mysqli_connect_error());
 $query=mysqli_query($db,"SELECT * FROM `mri_scan` WHERE 1 ") or die(mysqli_error($db));
$row1=mysqli_fetch_array($query);

$mri_val=$row1['mri_val'];
$report=$row1['report'];
$p_id=$row1['p_id'];
$d_id=$row1['d_id'];
if($report=="Tumor")
{
    $report="Brain Tumor Detected";
    
}
if(isset($_GET['back']))
{
echo ' <script language="javascript" type="text/javascript">
parent.document.location="prequest.php";
</script>';    
}


if(isset($_GET['send']))
{
    if($report=="Brain Tumor Detected")
    {
        $mes="Hello I have checked your MRI scan, Unfortunatly your MRI Scan have the presence of Brain Tumor, But dont worry about this disease, it is a curable One, We will start the treatement to cure it, contact me for more details";
    }
    else
    {
                $mes="Hello I have checked your MRI scan, Your scan is Normal and you do not have any sighs of Brain Tumor, so dont wory about anything";
    }
    $query1=mysqli_query($db, "INSERT INTO `chat` (`user_id`, `doctor_id`, `mes`, `ft`) VALUES ('$p_id','$d_id','$mes',2)") or die(mysqli_error($db));
    $query=mysqli_query($db,"UPDATE `send_mri` SET `status`=1 WHERE `patient_id`='$p_id' && `doctor_id`='$d_id'") or die(mysqli_error($db));
    
echo ' <script language="javascript" type="text/javascript">
alert("Reports are successfully sended to the Patient");
parent.document.location="dhome.php";
</script>';

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
  width: 70%;
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
}
</style> 
<meta http-equiv="refresh" content="7">
</head>
<body>

<center><h2>Details</h2></center>
<center>

<form method="GET" action="report_generate.php">
<div class="card">
  <img src=<?php echo "\"$mri_val\""; ?> alt="Avatar" style="width:100%">
  <div class="container">
    <h4><b><?php echo "Report : $report" ?></b></h4>
  <?php 
  if($report!="Loading")
  {
  echo"
   <button type=\"submit\" name=\"send\" class=\"registerbtn\">Send</button>
  <br><br>";
  }
  ?>
   <button type="submit" name="back" class="registerbtn">Back</button>
  <br><br>

  </div>
</div>
</form>
</center>
<br><br>
</body>
</html> 