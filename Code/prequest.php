<?php
 $db=mysqli_connect("localhost","id16188360_projects","Iamedotembedded@1","id16188360_project")or die(mysqli_connect_error());
if(isset($_GET['back']))
{
echo ' <script language="javascript" type="text/javascript">
parent.document.location="dhome.php";
</script>';

}
 $val='hai';
?>

<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {
  box-sizing: border-box;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
  width: 90%;
  border: 1px solid #ddd;
}

th, td {
  text-align: left;
  padding: 8px;
}

tr:nth-child(even){background-color: #f2f2f2}

#myInput {
  background-image: url('/css/searchicon.png');
  background-position: 10px 10px;
  background-repeat: no-repeat;
  width: 90%;
  font-size: 16px;
  padding: 12px 20px 12px 40px;
  border: 1px solid #ddd;
  margin-bottom: 12px;
}
/*
#myTable {
  border-collapse: collapse;
  width: 100%;
  border: 1px solid #ddd;
  font-size: 18px;
}

#myTable th, #myTable td {
  text-align: left;
  padding: 12px;
}

#myTable tr {
  border-bottom: 1px solid #ddd;
}

#myTable tr.header, #myTable tr:hover {
  background-color: #f1f1f1;
}*/
</style>
</head>
<body>
<img src="doctors.jpg" alt="Nature" class="responsive" width="100%" height="300">
<br><br>
<form method="GET" action="prequest.php">
<button type="submit" name="back" class="registerbtn">Back</button>
</form>

<center>
<h2>Patient List</h2>
</center>
<center>

<input type="text" id="myInput" onkeyup="myFunction()" placeholder="Search for Patient names.." title="Type in a name">
<div style="overflow-x:auto;">

<table id="myTable">
  <tr class="header">
    <th style="width:60%;">Name</th>
    <th style="width:40%;">Age</th>
    <th style="width:60%;">DOB</th>
    <th style="width:40%;">Address</th>
    <th style="width:40%;">Email</th>
    <th style="width:40%;">MRI SCAN</th>
    <th style="width:40%;">Generate Report</th>
  </tr>
      
         <?php  
             $query=mysqli_query($db,"SELECT * FROM `doctor_login` WHERE 1 ") or die(mysqli_error($db));
             $row=mysqli_fetch_array($query);
		        $id=$row['id'];
		        
				$query=mysqli_query($db,"SELECT * FROM `send_mri` WHERE `doctor_id`='$id' && `status`=0 ") or die(mysqli_error($db));
                $x=1;
while($row=mysqli_fetch_array($query))
{
    	$id=$row['id'];
   
	$patient_id=$row['patient_id'];
   	$doctor_id=$row['doctor_id'];
    $mri_scan=$row['mri_scan'];
    
 $query5=mysqli_query($db,"SELECT * FROM `patient_details` WHERE `id`='$patient_id' ") or die(mysqli_error($db));
while($row5=mysqli_fetch_array($query5))
{   
	$name=$row5['name'];
	$age=$row5['age'];
	$dob=$row5['dob'];
	$address=$row5['address'];
	$email=$row5['email'];
}	   
	
echo" <tr>
    <td>$name</td>
	   
	<td>$age</td>
    <td>$dob</td>
    <td>$address</td>
    
    <td>$email</td>
    
    <td>  <img src=\"$mri_scan\" alt=\"Nature\"></td>
   
 <td> <form method=\"GET\" action=\"prequest.php\"><input type=\"submit\" name=\"$id\" value=\"Generate Report\">
	</form>
	</td>
	</tr>";
	
	if((isset($_GET[$id])))
{
                 $query=mysqli_query($db,"SELECT * FROM `send_mri` WHERE `id`= '$id' ") or die(mysqli_error($db));
                 $row=mysqli_fetch_array($query);
		        $mri_scan1=$row['mri_scan'];
		        
$query=mysqli_query($db,"UPDATE `mri_scan` SET `mri_val`='$mri_scan1', `p_id`='$patient_id', `d_id`='$doctor_id', `report`='Loading' WHERE 1 ") or die(mysqli_error($db));
	  
echo ' <script language="javascript" type="text/javascript">
alert("You will be redirected to the Report page");
parent.document.location="report_generate.php";
</script>';

break;
}
}
?> 

       
  
</table>
</div>
</center>
<script>
function myFunction() {
  var input, filter, table, tr, td, i, txtValue;
  input = document.getElementById("myInput");
  filter = input.value.toUpperCase();
  table = document.getElementById("myTable");
  tr = table.getElementsByTagName("tr");
  for (i = 0; i < tr.length; i++) {
    td = tr[i].getElementsByTagName("td")[0];
    if (td) {
      txtValue = td.textContent || td.innerText;
      if (txtValue.toUpperCase().indexOf(filter) > -1) {
        tr[i].style.display = "";
      } else {
        tr[i].style.display = "none";
      }
    }       
  }
}
</script>

</body>
</html>
