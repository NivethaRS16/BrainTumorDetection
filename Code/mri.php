<?php
 $con=mysqli_connect("localhost","id16188360_projects","Iamedotembedded@1","id16188360_project")or die(mysqli_connect_error());
if(isset($_GET['back']))
{
echo ' <script language="javascript" type="text/javascript">
parent.document.location="search_chat2.php";
</script>';

}
if(isset($_POST['but_upload'])){
   $query=mysqli_query($con,"SELECT * FROM `patient_login` WHERE 1 ") or die(mysqli_error($con));
             $row=mysqli_fetch_array($query);
		        $id=$row['id'];
	         $query=mysqli_query($con,"SELECT * FROM `doctor_chat_id` WHERE 1 ") or die(mysqli_error($con));
             $row1=mysqli_fetch_array($query);
		        $did=$row1['id'];
	    
  $name = $_FILES['file']['name'];
  $target_dir = "upload/";
  $target_file = $target_dir . basename($_FILES["file"]["name"]);

  // Select file type
  $imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));

  // Valid file extensions
  $extensions_arr = array("jpg","jpeg","png","gif");

  // Check extension
  if( in_array($imageFileType,$extensions_arr) ){
 
    // Convert to base64 
    $image_base64 = base64_encode(file_get_contents($_FILES['file']['tmp_name']) );
    $image = 'data:image/'.$imageFileType.';base64,'.$image_base64;
    // Insert record
    //$query = "insert into images(image) values('".$image."')";
  //  $query = "insert into `send_mri` (`mri_scan`) values('".$target_file."')";
  
  
  //  $query = "INSERT INTO `send_mri`(`patient_id`,`doctor_id`,`mri_scan`) VALUES ('$id','$did'".$target_file."')";
//    mysqli_query($con,$query);
  $query = "INSERT INTO `send_mri`(`patient_id`, `doctor_id`, `mri_scan`,`status`) VALUES ($id,$did,'".$target_file."',0)";
  mysqli_query($con,$query);
  // Upload file
    move_uploaded_file($_FILES['file']['tmp_name'],$target_dir.$name);
  echo ' <script language="javascript" type="text/javascript">
  alert("MRI Image Sended successfully");
  parent.document.location="search_chat2.php";

</script>';
  }
  else
  {
    
  }
 
}
?>
<html>
    <body>
<img src="dc.jpg" alt="Nature" class="responsive" width="100%" height="300">
<br><br>
<form method="GET" action="mri.php">
<button type="submit" name="back" class="registerbtn">Back</button>
</form>
<br><br>
<center>
<form method="post" action="" enctype='multipart/form-data'>
  <input type='file' name='file' />
  <input type='submit' value='Send MRI' name='but_upload'>
</form>
</body>
</center>
</html>