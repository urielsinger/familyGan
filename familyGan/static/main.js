$( document ).ready(function() {
    console.log( "ready!" );
    function readURL(input, boxName) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function(e) {
                $('#'+boxName).css('background-image', 'url('+e.target.result +')').css('background-size', '100%').hide().fadeIn(650);
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    if(window.location.href.split("/").length < 4 && window.location.href.split("/")[3]){
        $('#imagePreview').attr('src', "upload_files/"+window.location.href.split("/")[3]);
    }

    $("#imageUpload1").change(function() {
        readURL(this,"imagePreview1");
    });
    $("#imageUpload2").change(function() {
        readURL(this,"imagePreview2");
    });

    $("#uploadBtn").click(function() {
        var data=new FormData()
          data.append('image1',$("#imageUpload1")[0].files[0])
          data.append('image2',$("#imageUpload2")[0].files[0])

          $.ajax({
              url:"/upload",
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
                  console.log("upload success")
              }
      })
    });
});