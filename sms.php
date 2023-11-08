<?php
//custom functions and configurations for twilio
require_once "fn.php";

if(!empty($_POST['phone']) && !empty($_POST['message'])){

  $phone = $_POST['phone'];
  $message = $_POST['message'];
  $attempt = sendSMS($message, $phone);
  dump($attempt);
}
?>
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
</head>
<body>

<h2>Report a pothole</h2>

<form action="" method="post">
<label for="numberSelect">Select a number:</label>
    <select id="numberSelect" onchange="displaySelectedNumber()">
        <option value="">Select a number</option>
        <option value="+91**********"> Road services</option>
        <!-- Add more options as needed -->
    </select>
    <br><br>
    <label for="enteredNumber">Phone number:</label>
    <input type="text" id="enteredNumber" name="phone" readonly>
    <!--<label for="">Phone number</label><br>
    <input type="text" name="phone" required>
      -->
    <br>
    <label for="">Location</label><br>
    <textarea name="message" id="" required></textarea>
    <label for="latitude">Latitude:</label>
        <input type="text" id="latitude" name="latitude" readonly>
        
    <label for="longitude">Longitude:</label>
        <input type="text" id="longitude" name="longitude" readonly>
        
        <br><br>
    <button type="button" onclick="showLocation()">Get Current Location</button>
        <br><br>
        
    <br><br>
    <input type="submit" value="Send">
</form>
<script>
    function displaySelectedNumber() {
        var selectBox = document.getElementById("numberSelect");
        var selectedNumber = selectBox.options[selectBox.selectedIndex].value;
        document.getElementById("enteredNumber").value = selectedNumber;
    }
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
</body>
</html>