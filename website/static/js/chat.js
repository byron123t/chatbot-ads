var counter = 0;
$("#gpt-button").click(function(){
    counter++;
    var question = $("#chat-input").val();
    let html_data = '';
    html_data += `
    <a class="list-group-item list-group-item-action d-flex gap-3 py-3">
      <img src="static/img/user.png" alt="twbs" width="64" height="64" class="rounded-circle flex-shrink-0">
      <div class="d-flex gap-2 w-100 justify-content-between">
        <div>
          <p class="mb-0 opacity-75">${question}</p>
        </div>
      </div>
    </a>
    `;
    $("#chat-input").val('');
    $("#list-group").append(html_data);
    let gpt_data = '';
    gpt_data += `
    <a class="list-group-item list-group-item-action d-flex gap-3 py-3">
      <img src="static/img/bot.png" alt="twbs" width="64" height="64" class="rounded-circle flex-shrink-0">
        <div class="d-flex gap-2 w-100 justify-content-between">
        <div>
        <p id="assistant${counter}" class="mb-0 opacity-75"></p>
        </div>
    </div>
    </a>
    `;
    $("#list-group").append(gpt_data);

    $.ajax({
        type: "POST",
        url: "/",
        data: {'prompt': question },
        xhrFields: {
            onprogress: function(data) {
                let curdata = data.currentTarget.response;
                console.log(curdata);
                console.log("assistant"+counter);
                $("#assistant"+counter).html(curdata);
            }
        },
        // success: function (data) {
        //     let gpt_data = '';
        //     gpt_data += `
        //     <a href="#" class="list-group-item list-group-item-action d-flex gap-3 py-3">
        //         <img src="https://digital-practice.ams3.cdn.digitaloceanspaces.com/static%2Fapp%2Fimg%2Fopenai-logo.png" alt="twbs" width="32" height="32" class="rounded-circle flex-shrink-0">
        //         <div class="d-flex gap-2 w-100 justify-content-between">
        //         <div>
        //         <p class="mb-0 opacity-75">${data}</p>
        //         </div>
        //     </div>
        //     </a>
        //     `;
        //     $("#list-group").append(gpt_data);
        // }
    });
});
