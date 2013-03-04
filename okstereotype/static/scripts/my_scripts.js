$('#ubtn')
    .click(function () {
        var btn = $(this)
        btn.button('loading')
        // setTimeout(function () {
        //     btn.button('reset')
        // }, 3000)
    });
$('#analyze_btn')
    .click(function () {
            var btn = $(this)
            btn.button('loading')
            var width = document.getElementById('analyze_btn').offsetWidth;
            var progress = setInterval(function() {
            var $bar = $('.bar');
            var $p_bar = $("#pbar");
            if ($bar.width()>=width) {
                clearInterval(progress);
                $bar.hide()
                $p_bar.hide()
                $('.progress').removeClass('active');
            } 
            else {
                $bar.width($bar.width()+width/20);
                console.log($bar.width());
                console.log($bar.width() + width/20)
            }
            // console.log($bar.width());
            $bar.text(Math.floor(($bar.width()+ width/20)*100/width) + "%");
        }, 1000);
        // setTimeout(function () {
        //     btn.button('reset')
        // }, 3000)
    });

// $(document).ready(function(){
//     var progress = setInterval(function() {
//     var $bar = $('.bar');
//     var $p_bar = $("#pbar");
//     if ($bar.width()==400) {
//         clearInterval(progress);
//         $p_bar.hide()
//         $('.progress').removeClass('active');
//     } else {
//         $bar.width($bar.width()+40);
//     }
//     $bar.text($bar.width()/4 + "%");
// }, 800);

// });​