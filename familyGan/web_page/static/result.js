$( document ).ready(function() {
    console.log( "ready!" );
    var gender_range = document.getElementById("gender_range")
    var headPose_yaw_range = document.getElementById("headPose_yaw_range")
    var headPose_roll_range = document.getElementById("headPose_roll_range")
    var age_kid_range = document.getElementById("age_kid_range")
    var age_middle_range = document.getElementById("age_middle_range")
    var age_young_range = document.getElementById("age_young_range")
    var age_old_range = document.getElementById("age_old_range")
    var glasses_range = document.getElementById("glasses_range")
    var smile_range = document.getElementById("smile_range")
    var anger_range = document.getElementById("anger_range")
    var sadness_range = document.getElementById("sadness_range")
    var contempt_range = document.getElementById("contempt_range")
    var disgust_range = document.getElementById("disgust_range")
    var fear_range = document.getElementById("fear_range")
    var happiness_range = document.getElementById("happiness_range")
    var neutral_range = document.getElementById("neutral_range")
    var surprise_range = document.getElementById("surprise_range")
    var eyeMakeup_range = document.getElementById("eyeMakeup_range")
    var lipMakeup_range = document.getElementById("lipMakeup_range")
    var beard_range = document.getElementById("beard_range")
    var facialhair_range = document.getElementById("facialhair_range")
    var moustache_range = document.getElementById("moustache_range")
    var sideburns_range = document.getElementById("sideburns_range")



    if(window.location.href.endsWith(".png")){
        $('.image--cover').attr('src', "get_child_image/"+window.location.href.split("/")[3]).hide().fadeIn(1000);
    }

    $(".overlay").css("display","none");

    $("#generateBtn").click(function() {
          var data=new FormData()
            data.append('child_path', window.location.href.split("/")[3]);
            data.append('gender',gender_range.value);
            data.append('headPose_yaw',headPose_yaw_range.value);
            data.append('headPose_roll',headPose_roll_range.value);
            data.append('age_kid',age_kid_range.value);
            data.append('age_middle',age_middle_range.value);
            data.append('age_young',age_young_range.value);
            data.append('age_old',age_old_range.value);
            data.append('glasses',glasses_range.value);
            data.append('smile',smile_range.value);
            data.append('anger',anger_range.value);
            data.append('sadness',sadness_range.value);
            data.append('contempt',contempt_range.value);
            data.append('disgust',disgust_range.value);
            data.append('fear',fear_range.value);
            data.append('happiness',happiness_range.value);
            data.append('neutral',neutral_range.value);
            data.append('surprise',surprise_range.value);
            data.append('eyeMakeup',eyeMakeup_range.value);
            data.append('lipMakeup',lipMakeup_range.value);
            data.append('beard',beard_range.value);
            data.append('facialhair',facialhair_range.value);
            data.append('moustache',moustache_range.value);
            data.append('sideburns',sideburns_range.value);

          $(".overlay").css("display","block");
          $.ajax({
              url:"/generate",
              type:'POST',
              data: data,
              cache:false,
              processData:false,
              contentType:false,
              error:function(xhr, status, error) {
                  console.log( xhr.responseText);
                  console.log("upload error")
              },
              success:function(data){
                  console.log(data)
                  $('.image--cover').attr('src', "get_child_image/"+data).hide().fadeIn(1000);
                  $(".overlay").css("display","none");
              }
      })


    });

    gender_range.oninput = function() {
        var gender_value = document.getElementById("gender_value");
        gender_value.innerHTML = this.value;
    }
    headPose_yaw_range.oninput = function() {
        var headPose_yaw_value = document.getElementById("headPose_yaw_value");
        headPose_yaw_value.innerHTML = this.value;
    }
    headPose_roll_range.oninput = function() {
        var headPose_roll_value = document.getElementById("headPose_roll_value");
        headPose_roll_value.innerHTML = this.value;
    }
    age_kid_range.oninput = function() {
        var age_kid_value = document.getElementById("age_kid_value");
        age_kid_value.innerHTML = this.value;
    }
    age_middle_range.oninput = function() {
        var age_middle_value = document.getElementById("age_middle_value");
        age_middle_value.innerHTML = this.value;
    }
    age_young_range.oninput = function() {
        var age_young_value = document.getElementById("age_young_value");
        age_young_value.innerHTML = this.value;
    }
    age_old_range.oninput = function() {
        var age_old_value = document.getElementById("age_old_value");
        age_old_value.innerHTML = this.value;
    }
    glasses_range.oninput = function() {
        var glasses_value = document.getElementById("glasses_value");
        glasses_value.innerHTML = this.value;
    }
    smile_range.oninput = function() {
        var smile_value = document.getElementById("smile_value");
        smile_value.innerHTML = this.value;
    }
    anger_range.oninput = function() {
        var anger_value = document.getElementById("anger_value");
        anger_value.innerHTML = this.value;
    }
    sadness_range.oninput = function() {
        var sadness_value = document.getElementById("sadness_value");
        sadness_value.innerHTML = this.value;
    }
    contempt_range.oninput = function() {
        var contempt_value = document.getElementById("contempt_value");
        contempt_value.innerHTML = this.value;
    }
    disgust_range.oninput = function() {
        var disgust_value = document.getElementById("disgust_value");
        disgust_value.innerHTML = this.value;
    }
    fear_range.oninput = function() {
        var fear_value = document.getElementById("fear_value");
        fear_value.innerHTML = this.value;
    }
    happiness_range.oninput = function() {
        var happiness_value = document.getElementById("happiness_value");
        happiness_value.innerHTML = this.value;
    }
    neutral_range.oninput = function() {
        var neutral_value = document.getElementById("neutral_value");
        neutral_value.innerHTML = this.value;
    }
    surprise_range.oninput = function() {
        var surprise_value = document.getElementById("surprise_value");
        surprise_value.innerHTML = this.value;
    }
    eyeMakeup_range.oninput = function() {
        var eyeMakeup_value = document.getElementById("eyeMakeup_value");
        eyeMakeup_value.innerHTML = this.value;
    }
    lipMakeup_range.oninput = function() {
        var lipMakeup_value = document.getElementById("lipMakeup_value");
        lipMakeup_value.innerHTML = this.value;
    }
    beard_range.oninput = function() {
        var beard_value = document.getElementById("beard_value");
        beard_value.innerHTML = this.value;
    }
    facialhair_range.oninput = function() {
        var facialhair_value = document.getElementById("facialhair_value");
        facialhair_value.innerHTML = this.value;
    }
    moustache_range.oninput = function() {
        var moustache_value = document.getElementById("moustache_value");
        moustache_value.innerHTML = this.value;
    }
    sideburns_range.oninput = function() {
        var sideburns_value = document.getElementById("sideburns_value");
        sideburns_value.innerHTML = this.value;
    }


});