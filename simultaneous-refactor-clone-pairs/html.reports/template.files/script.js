$(document).ready(function() {
	
	var timers = new Array();

	$(".child").mouseenter(function(e) {
		var dialogElement = $(this).children(".mappedimage").children(".informationdialog");
		var top1 = e.pageY; //$(this).offset().top - dialogElement.height();
		var left1 = e.pageX; //$(this).offset().left + $(this).width() - dialogElement.width();
		if (left1 +  dialogElement.width() > $(document).width()) {
			left1 = $(document).width() - dialogElement.width();
		}
		if (top1 +  dialogElement.height() > $(document).scrollTop() + $(window).height()) {
			top1 = $(document).scrollTop() + $(window).height() - dialogElement.height() - 50;
		}
		timers.push(setTimeout(function(){ 
			if (dialogElement.dispaly != "block")
				dialogElement.css({top: top1, left: left1, display: "block"}).show();
			}, 1000));
	});
	
	$(".child").mouseleave(function() {
		for (var i = 0; i < timers.length; i++) {
			clearTimeout(timers[i]);
		}
		var dialogElement = $(this).children(".mappedimage").children(".informationdialog");
		dialogElement.hide();
		
	});

	$(".expand").click(function() {
		if (!$(this).hasClass("rotate")) {
			hideUnder(this);
		} else {
			showUnder(this);
		}
	});
	
	hideUnder = function(obj) {
		var thisNestingLevel = getParentID(obj);
		$(obj).addClass("rotate");
		var elements = $(".child.ID" + thisNestingLevel + "> .LEFT  .expand");
		for (var i = 0; i < elements.length; i++)
			hideUnder(elements[i]);
		$(".child.ID" + thisNestingLevel).hide("fast");
		
	}
	
	showUnder = function(obj) {
		var thisNestingLevel = getParentID(obj);
		$(obj).removeClass("rotate");
		$(".child.ID" + thisNestingLevel).show("fast");
		var elements = $(".child.ID" + thisNestingLevel + "> .LEFT  .expand");
		for (var i = 0; i < elements.length; i++)
			showUnder(elements[i]);
		
	}
	
	getParentID = function(obj) {
		if (!$(obj).hasClass("rotate")) {
			return $(obj).attr("class").substring($(obj).attr("class").indexOf("ID") + 2);
		} else {
			var classNameWithoutRotate = $(obj).attr("class").replace(" rotate", "");
			return classNameWithoutRotate.substring(classNameWithoutRotate.indexOf("ID") + 2);
		}
	}
	
	$(".codeShow").click(function(event) {
		var id = $(this).attr("id").substring(0, $(this).attr("id").indexOf("Show"));
		event.preventDefault();
		showCode($("#" + id).html());
	});
	
	$("#closecode").click(function() {
		hideCode();
	});
	
	showCode = function(code) {
		$("#sourcecode textarea").val($("<div/>").html(code).text());
		$("#invisible").show();
		$("#sourcecode").show();
	}	
	
	hideCode = function() {
		$("#invisible").hide();
		$("#sourcecode").hide();
	}
	
	$(document).keydown(function(e) {
	    // ESCAPE key pressed
	    if (e.keyCode == 27) {
	    	hideCode();
	    }
	});
});
