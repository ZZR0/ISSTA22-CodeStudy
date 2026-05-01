from django.http import JsonResponse

def get_json_value(request):
    if 'hello' in request.GET:
        return JsonResponse({
            "status": "success",
            "message": "Hello, World! You passed the 'hello' parameter."
        })
    else:
        return JsonResponse({
            "status": "error",
            "message": "Query parameter '?hello' is missing."
        }, status=400)