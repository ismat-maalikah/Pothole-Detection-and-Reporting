<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
        }

        h2 {
            color: #333;
            margin-bottom: 20px;
        }

        form {
            background-color: #f7f7f7;
            padding: 20px;
            border-radius: 5px;
            max-width: 600px;
            margin: 0 auto;
        }

        label {
            color: #555;
            display: block;
            margin-bottom: 5px;
        }

        input[type="text"], textarea {
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 3px;
            font-size: 14px;
        }
        
        button[type="button"] {
            background-color: #007BFF;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        
        input[type="submit"] {
            background-color: #007BFF;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }

        input[type="submit"]:hover {
            background-color: #0056b3;
        }
    </style>

<script>
        function showLocation() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function(position) {
                    document.getElementById("latitude").value = position.coords.latitude;
                    document.getElementById("longitude").value = position.coords.longitude;
                                                                            });
        } else {
                alert("Geolocation is not supported by this browser.");
                }
                                }
    </script>


</head>
<body>

<h2>Report a pothole via email</h2>

<form action="" method="post"enctype="multipart/form-data" >
        <label for="name">Enter your name:</label>
        <textarea name="name" id="name" required></textarea>

        <label for="email">Enter an email:</label>
        <textarea name="email" id="email" required></textarea>

        <label for="file">Upload Image:</label>
        <input type="file" name="file" id="file">

        <label for="message">Location:</label>
        <textarea name="message" id="message" required></textarea>

        <label for="latitude">Latitude:</label>
        <input type="text" id="latitude" name="latitude" readonly>
        
        <label for="longitude">Longitude:</label>
        <input type="text" id="longitude" name="longitude" readonly>
        
        <br><br>
        <button type="button" onclick="showLocation()">Get Current Location</button>
        <br><br>

    <input type="submit" name="send" value="Send">
</form>


</body>
</html>
<?php

use PHPMailer\PHPMailer\PHPMailer;
use PHPMailer\PHPMailer\SMTP;
use PHPMailer\PHPMailer\Exception;

$name = $email = $message = $latitude = $longitude =$mailFrom= '';
$uploadfile =$mail= '';

if(isset($_POST['send']))
{
    $name=$_POST['name'] ?? '';
    $email=$_POST['email']?? '';
    $message=isset($_POST['message']) ? $_POST['message'] : ''; // Check if set before assigning
    $longitude = isset($_POST['longitude']) ? $_POST['longitude'] : '';
    $latitude = isset($_POST['latitude']) ? $_POST['latitude'] : '';
    $uploadfile = $_FILES['file']['tmp_name'];

//Load Composer's autoloader
require 'src\Exception.php';
require 'src\PHPMailer.php';
require 'src\SMTP.php';

//Create an instance; passing `true` enables exceptions
$mail = new PHPMailer(true);

try {
    //Server settings
    //$mail->SMTPDebug = SMTP::DEBUG_SERVER;                      //Enable verbose debug output
    $mail->isSMTP();                                            //Send using SMTP
    $mail->Host       = 'smtp.gmail.com';                     //Set the SMTP server to send through
    $mail->SMTPAuth   = true;                                   //Enable SMTP authentication
    $mail->Username   = '*************';                     //SMTP username
    $mail->Password   = '*************';                               //SMTP password
    $mail->SMTPSecure = PHPMailer::ENCRYPTION_SMTPS;            //Enable implicit TLS encryption
    $mail->Port       = 465;                                    //TCP port to connect to; use 587 if you have set `SMTPSecure = PHPMailer::ENCRYPTION_STARTTLS`
    $mail->SMTPOptions = array(
        'ssl' => array(
            'verify_peer' => false,
            'verify_peer_name' => false,
            'allow_self_signed' => true
        )
    );
    if (!empty($uploadfile) && file_exists($uploadfile)) {
        $mail->addAttachment($uploadfile, 'UploadedFile'); // Attach the uploaded file
    }

    //Recipients
    $mailFrom = $email;
    $mail->setFrom($mailFrom, 'mail form');
    $mail->addAddress('**authoritymail**', 'pothole mail');     //Add a recipient
    
    //Content
    $mail->isHTML(true);     //Set email format to HTML
    $mail->Subject = 'Pothole detected';
    $bodyContent = "name: $name<br>";

    $bodyContent = "Location: $message<br>";
        $bodyContent .= "Coordinates: Latitude - $latitude, Longitude - $longitude<br>";

        if (!empty($uploadfile) && file_exists($uploadfile)) {
            $mail->addStringEmbeddedImage(file_get_contents($uploadfile), 'UploadedFile', 'UploadedFile', 'base64', 'image/jpeg');
            $bodyContent .= 'Please see the attached image:<br><img src="cid:UploadedFile" alt="Uploaded Image">';
        }

        $mail->Body = $bodyContent;
    


    $mail->send();
    echo "<div class='success'> msg has been send </div>";
} catch (Exception $e) {
    echo "<div class='alert'>msg not send </div>";
}
}
?> 
