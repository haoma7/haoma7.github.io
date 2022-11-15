var swiper = new Swiper('.swiper-container', {
  coverflow: {
        rotate: 0,
        stretch: 10,
        depth: 150,
        scale:10,
        modifier: 1.5,
        slideShadows:true,
    },
    pagination: '.swiper-pagination',
    autoplay:true,
    speed:3000,
    paginationClickable: true,
    effect: 'coverflow',
    grabCursor:true,
    centeredSlides: true,
    slidesPerView: 'auto',
    autoHeight:true,
    initialSlide: 1,
    mousewheelControl: true,
    
    navigation:{
      nextEl: ".swiper-button-next",
      prevEl: ".swiper-button-prev"
    },

    keyboard: {
      enabled: true,
    },

} );

$(".swiper-container").hover(function() {
  swiper.stopAutoplay();
}, function() {
  swiper.startAutoplay();
});



$("body").keydown(function(e) {
  if(e.keyCode == 38 || e.keyCode == 37) { // top
    swiper.slidePrev();
  }
  else if(e.keyCode == 40||e.keyCode==39) { // bottom
    swiper.slideNext();
  }
});


